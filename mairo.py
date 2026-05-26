import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass

import cv2
import pygame

from hand_net import recv_packet, send_packet


SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
FPS = 60
TILE = 36
GRAVITY = 0.85
MOVE_SPEED = 5.0
JUMP_SPEED = -15.5

SKY = (100, 190, 255)
CLOUD = (245, 250, 255)
GROUND = (185, 112, 48)
GROUND_DARK = (120, 72, 32)
BRICK = (190, 90, 48)
QUESTION = (238, 174, 41)
PIPE = (42, 168, 75)
PIPE_DARK = (26, 107, 48)
COIN = (255, 221, 66)
PLAYER = (224, 55, 55)
PLAYER_HAT = (160, 24, 32)
ENEMY = (124, 76, 42)
TEXT = (24, 36, 48)
WHITE = (255, 255, 255)


@dataclass
class Controls:
    left: bool = False
    right: bool = False
    jump: bool = False
    source: str = "keyboard"
    gesture: str = "No hand"


class HandTracker:
    """MediaPipeから横移動とジャンプ入力を作る小さなアダプタ。"""

    def __init__(self, camera_index=0):
        import mediapipe as mp

        self.cap = cv2.VideoCapture(camera_index)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.debug_frame = None
        self.enabled = self.cap.isOpened()
        self.jump_latch = False

    def close(self):
        if self.cap:
            self.cap.release()
        self.hands.close()

    def update(self):
        controls = Controls()
        if not self.enabled:
            controls.gesture = "Camera not found"
            return controls

        ok, frame = self.cap.read()
        if not ok:
            controls.gesture = "Camera read failed"
            return controls

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            wrist = landmarks[0]
            index_tip = landmarks[8]

            # 画面を3分割して、手首より人差し指の横位置で左右移動を決める。
            if index_tip.x < 0.42:
                controls.left = True
            elif index_tip.x > 0.58:
                controls.right = True

            fingers_up = self._count_fingers_up(landmarks)
            jump_pose = fingers_up >= 4

            controls.jump = jump_pose and not self.jump_latch
            self.jump_latch = jump_pose
            controls.source = "hand"
            controls.gesture = self._gesture_name(controls, fingers_up)
        else:
            self.jump_latch = False

        cv2.putText(
            frame,
            controls.gesture,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )
        self.debug_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return controls

    def _count_fingers_up(self, landmarks):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        count = sum(landmarks[tip].y < landmarks[pip].y for tip, pip in zip(tips, pips))
        thumb_open = abs(landmarks[4].x - landmarks[9].x) > abs(landmarks[3].x - landmarks[9].x)
        return count + int(thumb_open)

    def _gesture_name(self, controls, fingers_up):
        direction = "Left" if controls.left else "Right" if controls.right else "Center"
        action = "Jump" if controls.jump else f"{fingers_up} fingers"
        return f"{direction} / {action}"

    def draw_debug(self, surface):
        if self.debug_frame is None:
            return
        preview = pygame.surfarray.make_surface(self.debug_frame.swapaxes(0, 1))
        preview = pygame.transform.smoothscale(preview, (192, 108))
        surface.blit(preview, (SCREEN_WIDTH - 204, 12))
        pygame.draw.rect(surface, WHITE, (SCREEN_WIDTH - 204, 12, 192, 108), 2)


class RemoteHandTracker:
    """ローカルのカメラ画像をリモートサーバに送り、操作結果だけ受け取る。"""

    def __init__(self, host, port, camera_index=0):
        self.host = host
        self.port = port
        self.cap = cv2.VideoCapture(camera_index)
        self.enabled = self.cap.isOpened()
        self.sock = None
        self.last_connect_try = 0
        self.debug_frame = None
        self.last_controls = Controls(gesture="Connecting to server")

    def close(self):
        if self.sock:
            self.sock.close()
        if self.cap:
            self.cap.release()

    def update(self):
        if not self.enabled:
            return Controls(gesture="Camera not found")

        ok, frame = self.cap.read()
        if not ok:
            return Controls(gesture="Camera read failed")

        frame = cv2.flip(frame, 1)
        if not self._ensure_connection():
            self._set_debug_frame(frame, self.last_controls.gesture)
            return self.last_controls

        try:
            small = cv2.resize(frame, (320, 240))
            ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            if not ok:
                return Controls(gesture="JPEG encode failed")

            send_packet(self.sock, encoded.tobytes())
            response = json.loads(recv_packet(self.sock).decode("utf-8"))
            controls = Controls(
                left=bool(response.get("left")),
                right=bool(response.get("right")),
                jump=bool(response.get("jump")),
                source="hand",
                gesture=response.get("gesture", "Remote hand"),
            )
            self.last_controls = controls
        except (OSError, ConnectionError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            self._disconnect()
            self.last_controls = Controls(gesture=f"Server lost: {exc}")

        self._set_debug_frame(frame, self.last_controls.gesture)
        return self.last_controls

    def _ensure_connection(self):
        if self.sock:
            return True

        now = time.monotonic()
        if now - self.last_connect_try < 2.0:
            return False
        self.last_connect_try = now

        try:
            sock = socket.create_connection((self.host, self.port), timeout=1.0)
            sock.settimeout(0.25)
            self.sock = sock
            self.last_controls = Controls(gesture=f"Remote {self.host}:{self.port}")
            return True
        except OSError as exc:
            self.last_controls = Controls(gesture=f"Server unavailable: {exc}")
            return False

    def _disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def _set_debug_frame(self, frame, gesture):
        cv2.putText(
            frame,
            gesture,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )
        self.debug_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def draw_debug(self, surface):
        if self.debug_frame is None:
            return
        preview = pygame.surfarray.make_surface(self.debug_frame.swapaxes(0, 1))
        preview = pygame.transform.smoothscale(preview, (192, 108))
        surface.blit(preview, (SCREEN_WIDTH - 204, 12))
        pygame.draw.rect(surface, WHITE, (SCREEN_WIDTH - 204, 12, 192, 108), 2)


class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 30, 42)
        self.vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.facing = 1
        self.invincible_timer = 0

    def update(self, controls, solids):
        if controls.left:
            self.vel.x = -MOVE_SPEED
            self.facing = -1
        elif controls.right:
            self.vel.x = MOVE_SPEED
            self.facing = 1
        else:
            self.vel.x *= 0.75
            if abs(self.vel.x) < 0.2:
                self.vel.x = 0

        if controls.jump and self.on_ground:
            self.vel.y = JUMP_SPEED
            self.on_ground = False

        self.vel.y = min(self.vel.y + GRAVITY, 18)
        self.rect.x += round(self.vel.x)
        self._collide(solids, "x")
        self.rect.y += round(self.vel.y)
        self.on_ground = False
        self._collide(solids, "y")
        self.invincible_timer = max(0, self.invincible_timer - 1)

    def _collide(self, solids, axis):
        for solid in solids:
            if not self.rect.colliderect(solid):
                continue
            if axis == "x":
                if self.vel.x > 0:
                    self.rect.right = solid.left
                elif self.vel.x < 0:
                    self.rect.left = solid.right
                self.vel.x = 0
            else:
                if self.vel.y > 0:
                    self.rect.bottom = solid.top
                    self.on_ground = True
                elif self.vel.y < 0:
                    self.rect.top = solid.bottom
                self.vel.y = 0

    def draw(self, surface, camera_x):
        x = self.rect.x - camera_x
        y = self.rect.y
        blink = self.invincible_timer and self.invincible_timer % 8 < 4
        if blink:
            return
        pygame.draw.rect(surface, PLAYER, (x, y + 10, 30, 32), border_radius=4)
        pygame.draw.rect(surface, PLAYER_HAT, (x + 2, y, 26, 14), border_radius=3)
        pygame.draw.rect(surface, (245, 184, 122), (x + 5, y + 14, 20, 15), border_radius=4)
        eye_x = x + 19 if self.facing > 0 else x + 9
        pygame.draw.circle(surface, TEXT, (eye_x, y + 20), 2)


class Enemy:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 32, 30)
        self.vel_x = -1.3
        self.alive = True

    def update(self, solids):
        if not self.alive:
            return
        self.rect.x += round(self.vel_x)
        hit_wall = False
        for solid in solids:
            if self.rect.colliderect(solid):
                hit_wall = True
                if self.vel_x > 0:
                    self.rect.right = solid.left
                else:
                    self.rect.left = solid.right
        below = self.rect.move(0, 4)
        has_floor = any(below.colliderect(solid) for solid in solids)
        if hit_wall or not has_floor:
            self.vel_x *= -1

    def draw(self, surface, camera_x):
        if not self.alive:
            return
        x = self.rect.x - camera_x
        y = self.rect.y
        pygame.draw.ellipse(surface, ENEMY, (x, y, 32, 30))
        pygame.draw.circle(surface, WHITE, (x + 10, y + 10), 5)
        pygame.draw.circle(surface, WHITE, (x + 22, y + 10), 5)
        pygame.draw.circle(surface, TEXT, (x + 10, y + 11), 2)
        pygame.draw.circle(surface, TEXT, (x + 22, y + 11), 2)


class Course:
    def __init__(self):
        self.solids = []
        self.coins = []
        self.enemies = []
        self.flag = pygame.Rect(3860, 130, 20, 300)
        self.goal = False
        self._build()

    def _add_block(self, x, y, w=1, h=1):
        for row in range(h):
            for col in range(w):
                self.solids.append(pygame.Rect(x + col * TILE, y + row * TILE, TILE, TILE))

    def _add_platform(self, x, y, w):
        self._add_block(x, y, w, 1)

    def _build(self):
        for start, length in [(0, 22), (900, 12), (1440, 18), (2360, 11), (2940, 29)]:
            self._add_block(start, SCREEN_HEIGHT - TILE, length, 2)

        self._add_platform(360, 370, 4)
        self._add_platform(560, 305, 3)
        self._add_platform(810, 250, 4)
        self._add_platform(1260, 340, 5)
        self._add_platform(1600, 280, 4)
        self._add_platform(1850, 220, 3)
        self._add_platform(2140, 345, 5)
        self._add_platform(2580, 290, 4)
        self._add_platform(3140, 330, 5)
        self._add_platform(3380, 260, 4)

        self._add_block(1080, SCREEN_HEIGHT - TILE * 3, 2, 2)
        self._add_block(1980, SCREEN_HEIGHT - TILE * 4, 2, 3)
        self._add_block(2820, SCREEN_HEIGHT - TILE * 3, 2, 2)
        for i in range(6):
            self._add_block(3640 + i * TILE, SCREEN_HEIGHT - TILE * (i + 2), 1, i + 1)

        coin_positions = [
            (410, 320),
            (590, 255),
            (850, 200),
            (1300, 290),
            (1640, 230),
            (1880, 170),
            (2180, 295),
            (2620, 240),
            (3180, 280),
            (3420, 210),
        ]
        self.coins = [pygame.Rect(x, y, 18, 18) for x, y in coin_positions]
        self.enemies = [
            Enemy(700, SCREEN_HEIGHT - TILE - 30),
            Enemy(1510, SCREEN_HEIGHT - TILE - 30),
            Enemy(2470, SCREEN_HEIGHT - TILE - 30),
            Enemy(3220, SCREEN_HEIGHT - TILE - 30),
        ]

    def update(self, player):
        for enemy in self.enemies:
            enemy.update(self.solids)
            if enemy.alive and player.rect.colliderect(enemy.rect):
                if player.vel.y > 0 and player.rect.bottom - enemy.rect.top < 18:
                    enemy.alive = False
                    player.vel.y = JUMP_SPEED * 0.55
                elif player.invincible_timer == 0:
                    player.rect.x = max(80, player.rect.x - 120)
                    player.vel.x = -5
                    player.vel.y = JUMP_SPEED * 0.45
                    player.invincible_timer = 90

        self.coins = [coin for coin in self.coins if not player.rect.colliderect(coin)]
        if player.rect.colliderect(self.flag):
            self.goal = True

    def draw(self, surface, camera_x):
        draw_background(surface, camera_x)
        for rect in self.solids:
            draw_tile(surface, rect.x - camera_x, rect.y)
        for coin in self.coins:
            x = coin.centerx - camera_x
            y = coin.centery
            pygame.draw.circle(surface, COIN, (x, y), 10)
            pygame.draw.circle(surface, (184, 130, 24), (x, y), 10, 2)
        for enemy in self.enemies:
            enemy.draw(surface, camera_x)
        self._draw_flag(surface, camera_x)

    def _draw_flag(self, surface, camera_x):
        x = self.flag.x - camera_x
        pygame.draw.rect(surface, (235, 235, 235), (x, self.flag.y, 8, self.flag.h))
        pygame.draw.polygon(surface, (42, 188, 88), [(x + 8, self.flag.y), (x + 88, self.flag.y + 28), (x + 8, self.flag.y + 56)])


def draw_background(surface, camera_x):
    surface.fill(SKY)
    for cx, cy, scale in [(120, 80, 1.0), (520, 120, 0.8), (980, 70, 1.1), (1600, 115, 0.9), (2500, 85, 1.0), (3350, 125, 0.9)]:
        x = cx - camera_x * 0.25
        while x < -160:
            x += 1400
        draw_cloud(surface, int(x), cy, scale)
    for x in range(-TILE, SCREEN_WIDTH + TILE, TILE):
        pygame.draw.rect(surface, (82, 202, 92), (x, SCREEN_HEIGHT - TILE * 2, TILE, TILE))


def draw_cloud(surface, x, y, scale):
    sizes = [(0, 18, 32), (28, 4, 38), (62, 16, 30), (30, 28, 46)]
    for ox, oy, radius in sizes:
        pygame.draw.circle(surface, CLOUD, (x + int(ox * scale), y + int(oy * scale)), int(radius * scale))


def draw_tile(surface, x, y):
    rect = pygame.Rect(x, y, TILE, TILE)
    pygame.draw.rect(surface, GROUND, rect)
    pygame.draw.rect(surface, GROUND_DARK, rect, 2)
    pygame.draw.line(surface, (220, 145, 68), (x + 6, y + 10), (x + TILE - 8, y + 10), 2)
    pygame.draw.line(surface, (220, 145, 68), (x + 10, y + 24), (x + TILE - 6, y + 24), 2)


def draw_hud(surface, font, controls, coins_left, won):
    pygame.draw.rect(surface, (255, 255, 255, 190), (14, 12, 360, 86), border_radius=6)
    surface.blit(font.render(f"Gesture: {controls.gesture}", True, TEXT), (26, 24))
    surface.blit(font.render(f"Coins left: {coins_left}", True, TEXT), (26, 52))
    surface.blit(font.render("Hand: left/center/right, open palm = jump", True, TEXT), (26, 78))
    if won:
        big = pygame.font.SysFont(None, 72)
        msg = big.render("COURSE CLEAR!", True, WHITE)
        shadow = big.render("COURSE CLEAR!", True, TEXT)
        rect = msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 35))
        surface.blit(shadow, rect.move(4, 4))
        surface.blit(msg, rect)


def keyboard_controls(keys):
    return Controls(
        left=keys[pygame.K_LEFT] or keys[pygame.K_a],
        right=keys[pygame.K_RIGHT] or keys[pygame.K_d],
        jump=keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w],
        source="keyboard",
        gesture="Keyboard",
    )


def merge_controls(hand_controls, key_controls):
    return Controls(
        left=hand_controls.left or key_controls.left,
        right=hand_controls.right or key_controls.right,
        jump=hand_controls.jump or key_controls.jump,
        source=hand_controls.source if hand_controls.source == "hand" else key_controls.source,
        gesture=hand_controls.gesture if hand_controls.source == "hand" else key_controls.gesture,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe Hand Mario Course")
    parser.add_argument(
        "--control",
        choices=["local", "remote"],
        default="local",
        help="local: このPCでMediaPipe処理 / remote: サーバでMediaPipe処理",
    )
    parser.add_argument("--server-host", default="127.0.0.1", help="remote時のサーバIPまたはホスト名")
    parser.add_argument("--server-port", type=int, default=5005, help="remote時のサーバポート")
    parser.add_argument("--camera", type=int, default=0, help="ローカルPCのカメラ番号")
    return parser.parse_args()


def main():
    args = parse_args()
    pygame.init()
    pygame.display.set_caption("MediaPipe Hand Mario Course")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    if args.control == "remote":
        tracker = RemoteHandTracker(args.server_host, args.server_port, args.camera)
    else:
        tracker = HandTracker(camera_index=args.camera)
    course = Course()
    player = Player(80, SCREEN_HEIGHT - TILE * 3)
    camera_x = 0

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    course = Course()
                    player = Player(80, SCREEN_HEIGHT - TILE * 3)

            hand_controls = tracker.update()
            key_controls = keyboard_controls(pygame.key.get_pressed())
            controls = merge_controls(hand_controls, key_controls)

            if not course.goal:
                player.update(controls, course.solids)
                course.update(player)

            if player.rect.top > SCREEN_HEIGHT:
                player = Player(80, SCREEN_HEIGHT - TILE * 3)

            target_camera = player.rect.centerx - SCREEN_WIDTH // 2
            camera_x += (max(0, target_camera) - camera_x) * 0.12
            camera_x = min(camera_x, 3220)

            course.draw(screen, camera_x)
            player.draw(screen, camera_x)
            draw_hud(screen, font, controls, len(course.coins), course.goal)
            tracker.draw_debug(screen)

            pygame.display.flip()
            clock.tick(FPS)
    finally:
        tracker.close()
        pygame.quit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        cv2.destroyAllWindows()
        sys.exit(0)
