import json
import time
import uuid
import base64
import logging
import math

from numpy import linalg
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Any

import cv2
import mediapipe as mp
import numpy as np

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.core.utils import TempFileResponse

router = APIRouter(prefix="/ws", tags=["Websockets"])

NEW_WIDTH = 128
NEW_HEIGHT = 128
DESIRED_FPS = 30

# currently active connections. Currently stable and working fine.
# Use Redis in case the application scales and set gunicorn workers to more than 1
connections = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websockets")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=False,
    enable_segmentation=False,
)
mp_draw = mp.solutions.drawing_utils


def get_points(skelet: NamedTuple, point: int) -> list:
    return [
        skelet.pose_landmarks.landmark[point].x,
        skelet.pose_landmarks.landmark[point].y,
        skelet.pose_landmarks.landmark[point].z
    ]


def make_vec(first_point: list, second_point: list) -> np.array:
    return np.array(list(map(lambda x, y: x - y, first_point, second_point)))


def count_angle(fist_vec: np.array, second_vec: np.array) -> Any:
    cos = np.dot(fist_vec, second_vec) / (linalg.norm(fist_vec) * linalg.norm(second_vec))
    return math.degrees(math.acos(cos))


# Функция для вычисления угла между двумя векторами
def calculate_angle(a, b, c):
    a = np.array(a)  # Первая точка
    b = np.array(b)  # Вершина угла
    c = np.array(c)  # Вторая точка

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def calculate_distance_points(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def is_front_facing(l_hip: list[float], r_hip: list[float]):
    z_diff = abs(l_hip[2] - r_hip[2])  # Разница по координате z
    return z_diff < 0.1  # Если разница по глубине (z) мала, человек стоит лицом к камере


def draw_landmarks(frame, skelet):
    mp_draw.draw_landmarks(frame, skelet.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    for idx, lm in enumerate(skelet.pose_landmarks.landmark):
        h, w, c = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (int(cx), int(cy)), 10, (255, 0, 0), cv2.FILLED)


def show_fps(frame, p_time):
    current_time = time.time()
    fps = 1 / (current_time - p_time)
    p_time = current_time

    frame = cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    return fps, p_time


def process_high_knees(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        point_30_y = results.pose_landmarks.landmark[30].y
        point_29_y = results.pose_landmarks.landmark[29].y
        point_25_y = results.pose_landmarks.landmark[25].y
        point_26_y = results.pose_landmarks.landmark[26].y
        point_15_y = results.pose_landmarks.landmark[15].y
        point_16_y = results.pose_landmarks.landmark[16].y
        point_13_y = results.pose_landmarks.landmark[13].y
        point_14_y = results.pose_landmarks.landmark[14].y

        if (
                (point_30_y < point_25_y or point_29_y < point_26_y)
                and (point_15_y < point_13_y and point_16_y < point_14_y)
                and not jump_started
        ):
            jump_started = True
            repetitions_count += 1

        elif point_30_y >= point_25_y and point_29_y >= point_26_y:
            jump_started = False

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for _, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - p_time)
    p_time = current_time

    frame = cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


def process_jumping_jacks(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        left_shoulder_y = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ].y
        left_hand_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_shoulder_y = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ].y
        right_hand_y = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.RIGHT_WRIST
        ].y
        left_ankle_y = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.LEFT_ANKLE
        ].y
        right_ankle_y = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ].y

        if (
                left_hand_y > left_shoulder_y
                and right_hand_y > right_shoulder_y
                and not jump_started
        ):
            if (
                    left_hand_y > left_shoulder_y
                    and right_hand_y > right_shoulder_y
                    and left_ankle_y > right_ankle_y
                    and not jump_started
            ):
                jump_started = True
                repetitions_count += 1
        elif (
                left_hand_y <= left_shoulder_y
                and right_hand_y <= right_shoulder_y
                and left_ankle_y <= right_ankle_y
        ):
            jump_started = False

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for idx, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (int(cx), int(cy)), 10, (255, 0, 0), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - p_time)
    p_time = current_time

    frame = cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Боковой выпад
def side_lunge(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Ступни
        right_foot = get_points(results, 28)
        left_foot = get_points(results, 27)

        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)

        # Опора, вектор от левой ступни до правой
        support_left_vec = make_vec(right_foot[:2], left_foot[:2])
        # Опора, вектор от правой ступни до левой
        support_right_vec = make_vec(left_foot[:2], right_foot[:2])
        # Левая нога, вектор от левой ступни до левого бедра
        left_foot_vec = make_vec(left_hip[:2], left_foot[:2])
        # Правая нога, вектор от правой ступни до правого бедра
        right_foot_vec = make_vec(right_hip[:2], right_foot[:2])

        left_angle = count_angle(support_left_vec, left_foot_vec)
        right_angle = count_angle(support_right_vec, right_foot_vec)

        if (30 <= left_angle <= 60 or 30 <= right_angle <= 60) and not jump_started:
            jump_started = True
            repetitions_count += 1
        elif (left_angle < 30 or left_angle > 60) and (right_angle < 30 or right_angle > 60):
            jump_started = False

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Боковой удар
def side_kick(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    done = False
    if results.pose_landmarks:
        # Кисти
        right_wrist = get_points(results, 16)
        left_wrist = get_points(results, 15)

        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)

        # Глаза
        right_eye = get_points(results, 5)
        left_eye = get_points(results, 2)

        # Рот
        mouth = get_points(results, 10)

        if \
                round(left_eye[2], 2) == round(right_eye[2], 2) or \
                        round(left_eye[2], 2) + 0.01 == round(right_eye[2], 2) or \
                        round(left_eye[2], 2) == round(right_eye[2], 2) + 0.01:
            # Значит человек смотрит лицом в камеру
            if (right_hip[2] != left_hip[2]) and \
                    (
                            (right_wrist[1] - 0.11 <= mouth[1] and right_wrist[2] < mouth[2]) or
                            (left_wrist[1] - 0.11 <= mouth[1] and left_wrist[2] < mouth[2])
                    ):
                # Проверяем поворот бедра, и поднятие запястий выше или на уровень рта,
                # а так же то, что они ближе к камере чем рот
                done = True
        else:
            # Человек смотрит в бок
            if right_hip[2] < left_hip[2]:
                # Человек смотрит влево
                if (right_hip[0] != left_hip[0]) and \
                        (
                                (right_wrist[1] - 0.11 <= mouth[1] and right_wrist[0] > mouth[0]) or
                                (left_wrist[1] - 0.11 <= mouth[1] and left_wrist[0] > mouth[0])
                        ):
                    done = True
            else:
                # Человек смотрит вправо
                if (right_hip[0] != left_hip[0]) and \
                        (
                                (right_wrist[1] - 0.11 <= mouth[1] and right_wrist[0] < mouth[0]) or
                                (left_wrist[1] - 0.11 <= mouth[1] and left_wrist[0] < mouth[0])
                        ):
                    done = True
        if done and not jump_started:
            jump_started = True
            repetitions_count += 1
        elif not done:
            jump_started = False

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Велосипед
def bycicle(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Колени
        right_knee = get_points(results, 26)
        left_knee = get_points(results, 25)

        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)

        # Плечи
        right_shoulder = get_points(results, 12)
        left_shoulder = get_points(results, 11)

        # Левая часть туловища, вектор от левого бедра до левого плеча
        left_side_vec = make_vec(left_shoulder, left_hip)
        # Правая часть туловища, вектор от правого бедра до правого плеча
        right_side_vec = make_vec(right_shoulder, right_hip)
        # Левая нога, вектор от левого бедра до левого колена
        left_leg_vec = make_vec(left_knee, left_hip)
        # Правая нога, вектор от правого колена до правого бедра
        right_leg_vec = make_vec(right_knee, right_hip)

        # Высчитываем угол между вектором от бедра до колена и вектором от бедра до плеча
        left_angle = count_angle(left_leg_vec, left_side_vec)
        right_angle = count_angle(right_leg_vec, right_side_vec)

        # Условие if для начала повторения
        if (left_angle <= 90 or right_angle <= 90) and (
                left_knee[1] > right_knee[1] + 0.3 or right_knee[1] > left_knee[1] + 0.3) and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif (left_angle > 90 and right_angle > 90) or (
                abs(left_knee[1] - right_knee[1]) <= 0.1) and jump_started:
            repetitions_count += 1
            jump_started = False

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Махи ногами со скручиванием
def leg_swings(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Координаты левого бедра, колена и лодыжки
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Координаты правого бедра, колена и лодыжки
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Координаты левого и правого плеча
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        # Координаты левого и правого локтя
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

        # Координаты левой и правой кисти
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Вычисление углов
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Условие if для начала повторения
        if ((left_leg_angle > 45 and right_arm_angle < 45) or (
                right_leg_angle > 45 and left_arm_angle < 45)) and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and (
                (left_leg_angle <= 45 and right_arm_angle >= 45) or (right_leg_angle <= 45 and left_arm_angle >= 45)):
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Мельница
def melnica(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Координаты рук, плеч и лодыжек
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Вычисление угла между руками (угол в плечевом суставе через плечи)
        arms_angle = calculate_angle(left_wrist, left_shoulder, right_wrist)

        # Проверка условия, что руки образуют прямую линию и одна из рук рядом с ногой
        if 170 <= arms_angle <= 180:
            # Проверка, что правая рука рядом с правой ногой
            right_distance = calculate_distance(right_wrist, right_ankle)
            left_distance = calculate_distance(left_wrist, left_ankle)

            if right_distance < 0.1 or left_distance < 0.1 and not jump_started:
                jump_started = True

        elif jump_started:
            right_distance = calculate_distance(right_wrist, right_ankle)
            left_distance = calculate_distance(left_wrist, left_ankle)
            if right_distance > 0.1 and left_distance > 0.1:
                repetitions_count += 1
                jump_started = False

        draw_landmarks(frame, results)

    # Подсчет FPS
    current_time = time.time()
    fps = 1 / (current_time - p_time)
    p_time = current_time

    frame = cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Отведение ноги назад
def leg_abduption(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Глаза
        right_eye = get_points(results, 5)
        left_eye = get_points(results, 2)
        # Ступни
        right_foot = get_points(results, 28)
        left_foot = get_points(results, 27)
        # Колени
        right_knee = get_points(results, 26)
        left_knee = get_points(results, 25)
        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)
        # Плечи
        right_shoulder = get_points(results, 12)
        left_shoulder = get_points(results, 11)

        # Левая часть туловища, вектор от левого бедра до левого плеча
        left_side_vec = make_vec(left_shoulder, left_hip)
        # Правая часть туловища, вектор от правого бедра до правого плеча
        right_side_vec = make_vec(right_shoulder, right_hip)
        # Левая нога, вектор от левого бедра до левого колена
        left_leg_vec = make_vec(left_knee, left_hip)
        # Правая нога, вектор от правого бедра до правого колена
        right_leg_vec = make_vec(right_knee, right_hip)

        # Высчитываем угол между вектором от бедра до колена и вектором от бедра до плеча
        left_angle = count_angle(left_side_vec, left_leg_vec)
        right_angle = count_angle(right_side_vec, right_leg_vec)

        done = False
        if left_eye[0] > left_shoulder[0]:
            # Смотрим влево
            statement = (60 <= left_angle <= 90 and right_angle >= 95 and
                         right_foot[0] < left_foot[0]) or \
                        (60 <= right_angle <= 90 and left_angle >= 95 and
                         left_foot[0] < right_foot[0])
            if statement:
                done = True
        else:
            statement = (60 <= left_angle <= 90 and right_angle >= 95 and
                         right_foot[0] > left_foot[0]) or \
                        (60 <= right_angle <= 90 and left_angle >= 95 and
                         left_foot[0] > right_foot[0])
            if statement:
                done = True

        if done and not jump_started:
            jump_started = True
            repetitions_count += 1
        elif not done:
            jump_started = False

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Отжимания
def pushups(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Координаты левого плеча, локтя и запястья
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        # Координаты правого плеча, локтя и запястья
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        # Координаты левого бедра, колена и лодыжки
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        # Координаты правого бедра, колена и лодыжки
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        # Вычисление углов
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Засчитываем повтор при опускании туловища к полу (углы меньше 90 градусов)
        done = False
        # Условие if для начала повторения
        if (left_arm_angle < 90 or right_arm_angle < 90) and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and (left_arm_angle >= 90 and right_arm_angle >= 90):
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Планка вверх-вниз
def move_plank(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Глаза
        right_eye = get_points(results, 5)
        left_eye = get_points(results, 2)

        # Ступни
        right_foot = get_points(results, 28)
        left_foot = get_points(results, 27)

        # Колени
        right_knee = get_points(results, 26)
        left_knee = get_points(results, 25)

        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)

        # Плечи
        right_shoulder = get_points(results, 12)
        left_shoulder = get_points(results, 11)

        # Локти
        right_elbow = get_points(results, 14)
        left_elbow = get_points(results, 13)

        # Кисти
        right_wrist = get_points(results, 16)
        left_wrist = get_points(results, 15)

        # Левая часть туловища, вектор от левого бедра до левого плеча
        left_side_vec = make_vec(left_shoulder, left_hip)
        # Правая часть туловища, вектор от правого бедра до правого плеча
        right_side_vec = make_vec(right_shoulder, right_hip)
        # Левая нога, вектор от левого бедра до левого колена
        left_leg_vec = make_vec(left_knee, left_hip)
        # Правая нога, вектор от правого бедра до правого колена
        right_leg_vec = make_vec(right_knee, right_hip)
        # Левая нога, вектор от левой колена до левого ступни
        left_foot_vec = make_vec(left_foot, left_knee)
        # Правая нога, вектор от правой колена до правого ступни
        right_foot_vec = make_vec(right_foot, right_knee)
        # Векторы от плеча до локтя
        left_arm_vec = make_vec(left_elbow, left_shoulder)
        right_arm_vec = make_vec(right_elbow, right_shoulder)
        # Векторы от локтя до кисти
        left_hand_vec = make_vec(left_wrist, left_elbow)
        right_hand_vec = make_vec(right_wrist, right_elbow)

        # Высчитываем угол между вектором от бедра до колена и вектором от бедра до плеча
        left_angle = count_angle(left_side_vec, left_leg_vec)
        right_angle = count_angle(right_side_vec, right_leg_vec)
        # Вычисляем угол для проверки сгинания ног
        left_foot_angle = count_angle((-1) * left_leg_vec, left_foot_vec)
        right_foot_angle = count_angle((-1) * right_leg_vec, right_foot_vec)
        # Вычисляем угол между туловищем и руками
        left_arm_angle = count_angle((-1) * left_side_vec, left_arm_vec)
        right_arm_angle = count_angle((-1) * right_side_vec, right_arm_vec)
        # Вычисляем угол для проверки сгинания локтя
        left_hand_angle = count_angle((-1) * left_arm_vec, left_hand_vec)
        right_hand_angle = count_angle((-1) * right_arm_vec, right_hand_vec)

        statement_body = (left_angle > 90.0) or (right_angle > 90.0)
        statement_foot = (left_foot_angle > 90.0) or (right_foot_angle > 90.0)
        statement_arm = (60 <= left_arm_angle <= 100.0) or (60 <= right_arm_angle <= 100.0)
        statement_hand_start = (30 <= left_hand_angle <= 90.0) or (30 <= right_hand_angle <= 90.0)
        statement_hand_end = (left_hand_angle > 100.0) or (right_hand_angle > 100.0)
        # Условие if для начала повторения
        if not jump_started and statement_foot and statement_arm and statement_hand_end:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and statement_foot and statement_arm and statement_hand_start:
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Планка
def plank(frame, session_data: dict):
    if "jump_started" not in session_data or "start_time" not in session_data or "timer" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "timer": 0, "p_time": 0, "start_time": time.time()}
        )

    jump_started, repetitions_count, p_time, start_time, timer = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
        session_data["start_time"],
        session_data["timer"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Глаза
        right_eye = get_points(results, 5)
        left_eye = get_points(results, 2)
        # Ступни
        right_foot = get_points(results, 28)
        left_foot = get_points(results, 27)
        # Колени
        right_knee = get_points(results, 26)
        left_knee = get_points(results, 25)
        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)
        # Плечи
        right_shoulder = get_points(results, 12)
        left_shoulder = get_points(results, 11)
        # Локти
        right_elbow = get_points(results, 14)
        left_elbow = get_points(results, 13)
        # Кисти
        right_wrist = get_points(results, 16)
        left_wrist = get_points(results, 15)

        # Левая часть туловища, вектор от левого бедра до левого плеча
        left_side_vec = make_vec(left_shoulder, left_hip)
        # Правая часть туловища, вектор от правого бедра до правого плеча
        right_side_vec = make_vec(right_shoulder, right_hip)
        # Левая нога, вектор от левого бедра до левого колена
        left_leg_vec = make_vec(left_knee, left_hip)
        # Правая нога, вектор от правого бедра до правого колена
        right_leg_vec = make_vec(right_knee, right_hip)
        # Левая нога, вектор от левой колена до левого ступни
        left_foot_vec = make_vec(left_foot, left_knee)
        # Правая нога, вектор от правой колена до правого ступни
        right_foot_vec = make_vec(right_foot, right_knee)
        # Векторы от плеча до локтя
        left_arm_vec = make_vec(left_elbow, left_shoulder)
        right_arm_vec = make_vec(right_elbow, right_shoulder)
        # Векторы от локтя до кисти
        left_hand_vec = make_vec(left_wrist, left_elbow)
        right_hand_vec = make_vec(right_wrist, right_elbow)

        # Высчитываем угол между вектором от бедра до колена и вектором от бедра до плеча
        left_angle = count_angle(left_side_vec, left_leg_vec)
        right_angle = count_angle(right_side_vec, right_leg_vec)
        # Вычисляем угол для проверки сгинания ног
        left_foot_angle = count_angle((-1) * left_leg_vec, left_foot_vec)
        right_foot_angle = count_angle((-1) * right_leg_vec, right_foot_vec)
        # Вычисляем угол между туловищем и руками
        left_arm_angle = count_angle((-1) * left_side_vec, left_arm_vec)
        right_arm_angle = count_angle((-1) * right_side_vec, right_arm_vec)
        # Вычисляем угол для проверки сгинания локтя
        left_hand_angle = count_angle((-1) * left_arm_vec, left_hand_vec)
        right_hand_angle = count_angle((-1) * right_arm_vec, right_hand_vec)

        statement_body = left_angle > 100 or right_angle > 100
        statement_foot = (left_foot_angle > 90.0) or (right_foot_angle > 90.0)
        statement_arm = (60 <= left_arm_angle <= 100) or (60 <= right_arm_angle <= 100)
        statement_hand = (30 <= left_hand_angle <= 100) or (30 <= right_hand_angle <= 100)
        if statement_body and statement_foot and statement_arm and statement_hand and not jump_started:
            jump_started = True
            start_time = time.time()
        elif statement_body and statement_foot and statement_arm and statement_hand and jump_started:
            timer += time.time() - start_time  # Время выполнения приседа
            if round(timer, 0) > repetitions_count:
                repetitions_count += 1
            start_time = time.time()
            jump_started = True
        else:
            jump_started = False

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "timer": timer,
            "p_time": p_time,
            "start_time": start_time
        }
    )

    return frame, fps, repetitions_count


# Шаги скалолаза
def climbers_steps(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        # Рассчитываем угол в локте
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        # Рассчитываем угол в бедре
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)

        # Повторяем для правой стороны
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

        done = False
        # Условие if для начала повторения
        if ((r_elbow_angle > 160 and r_hip_angle < 90) or (
                l_elbow_angle > 160 and l_hip_angle < 90)) and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and (
                (r_elbow_angle <= 160 and r_hip_angle >= 90) or (l_elbow_angle <= 160 and l_hip_angle >= 90)):
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Удар ногой
def kick(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Получаем координаты необходимых точек
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        # Проверка, что руки выше пояса
        left_hand_above_hip = (left_elbow[1] < left_hip[1]) and (left_wrist[1] < left_hip[1])
        right_hand_above_hip = (right_elbow[1] < right_hip[1]) and (right_wrist[1] < right_hip[1])

        # Проверка выполнения удара ногой до параллели с полом
        left_foot_height = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        right_foot_height = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        left_hip_height = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_height = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

        # Проверка выполнения повторения
        # Условие if для начала повторения
        if (left_hand_above_hip and right_hand_above_hip) and (
                (left_foot_height < left_hip_height + 0.1) or (right_foot_height < right_hip_height + 0.1)
        ) and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and (
                (left_foot_height >= left_hip_height + 0.1 and right_foot_height >= right_hip_height + 0.1)
        ):
            jump_started = False
            repetitions_count += 1

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for idx, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (int(cx), int(cy)), 10, (255, 0, 0), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - p_time)
    p_time = current_time

    frame = cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Скручивания стоя
def standing_curls(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Получаем координаты необходимых точек
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

        # Рассчитываем расстояние между левым коленом и правым локтем, а также между правым коленом и левым локтем
        distance_left_knee_right_elbow = calculate_distance(left_knee, right_elbow)
        distance_right_knee_left_elbow = calculate_distance(right_knee, left_elbow)

        # Задаем пороговое значение для засчитывания упражнения
        threshold = 0.1  # Вы можете настроить это значение в зависимости от вашего видео

        # Проверка выполнения упражнения
        if (
                distance_left_knee_right_elbow < threshold or distance_right_knee_left_elbow < threshold) and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and (
                distance_left_knee_right_elbow >= threshold and distance_right_knee_left_elbow >= threshold):
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Подъем таза
def pelvic_lift(frame, session_data: dict, timing=False):
    if "jump_started" not in session_data or "start_time" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0, "start_time": time.time()}
        )

    jump_started, repetitions_count, p_time, start_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
        session_data["start_time"]
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Получаем координаты необходимых точек
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Проверка, что голова, руки и ступни прижаты к полу (y-координата ниже бедер)
        head_on_floor = left_shoulder[1] > left_hip[1] and right_shoulder[1] > right_hip[1]
        hands_on_floor = left_wrist[1] > left_hip[1] and right_wrist[1] > right_hip[1]
        feet_on_floor = left_ankle[1] > left_knee[1] and right_ankle[1] > right_knee[1]

        # Проверка угла поднятия таза (образование прямой линии от коленей до шеи с допустимым отклонением)
        hip_angle_left = calculate_angle(left_knee, left_hip, left_shoulder)
        hip_angle_right = calculate_angle(right_knee, right_hip, right_shoulder)

        # Условие для угла таза (прямой линии от коленей до шеи с допустимым отклонением)
        correct_hip_angle_left = 160 <= hip_angle_left <= 180
        correct_hip_angle_right = 160 <= hip_angle_right <= 180

        # Проверка выполнения упражнения
        # Обычный подъем таза подсчет повторений
        # Условие if для начала повторения
        if head_on_floor and hands_on_floor and feet_on_floor and (
                correct_hip_angle_left or correct_hip_angle_right) and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and not (correct_hip_angle_left or correct_hip_angle_right):
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
            "start_time": start_time
        }
    )

    return frame, fps, repetitions_count


# Удержание таза
def pelvic_static(frame, session_data: dict):
    if "jump_started" not in session_data or "start_time" not in session_data or "timer" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "timer": 0, "p_time": 0, "start_time": time.time()}
        )

    jump_started, repetitions_count, p_time, start_time, timer = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
        session_data["start_time"],
        session_data["timer"]
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Получаем координаты необходимых точек
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Проверка, что голова, руки и ступни прижаты к полу (y-координата ниже бедер)
        head_on_floor = left_shoulder[1] > left_hip[1] and right_shoulder[1] > right_hip[1]
        hands_on_floor = left_wrist[1] > left_hip[1] and right_wrist[1] > right_hip[1]
        feet_on_floor = left_ankle[1] > left_knee[1] and right_ankle[1] > right_knee[1]

        # Проверка угла поднятия таза (образование прямой линии от коленей до шеи с допустимым отклонением)
        hip_angle_left = calculate_angle(left_knee, left_hip, left_shoulder)
        hip_angle_right = calculate_angle(right_knee, right_hip, right_shoulder)

        # Условие для угла таза (прямой линии от коленей до шеи с допустимым отклонением)
        correct_hip_angle_left = 140 <= hip_angle_left <= 180
        correct_hip_angle_right = 140 <= hip_angle_right <= 180

        # Проверка выполнения упражнения
        # Удержание таза подсчет времени
        if head_on_floor and hands_on_floor and feet_on_floor and \
                (correct_hip_angle_left or correct_hip_angle_right) and not jump_started:
            jump_started = True
            start_time = time.time()
        elif head_on_floor and hands_on_floor and feet_on_floor and (
                correct_hip_angle_left or correct_hip_angle_right) and jump_started:
            timer += time.time() - start_time  # Время выполнения приседа
            if round(timer, 0) > repetitions_count:
                repetitions_count += 1
            start_time = time.time()
        else:
            jump_started = False
        draw_landmarks(frame, results)
    fps, p_time = show_fps(frame, p_time)
    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "timer": timer,
            "p_time": p_time,
            "start_time": start_time
        }
    )
    return frame, fps, repetitions_count


# Подъем ног с упором на локти
def leg_raises_elbow_rest(frame, session_data: dict):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Получаем координаты необходимых точек
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        # Проверка, что локти находятся на полу (ниже плеч)
        elbows_on_floor = (left_elbow[1] > left_shoulder[1]) and (right_elbow[1] > right_shoulder[1])

        # Проверка высоты подъема ног (ноги должны быть выше бедер)
        legs_raised = (left_ankle[1] + 0.2 < left_hip[1]) and (right_ankle[1] + 0.2 < right_hip[1])

        # Проверка выполнения упражнения
        if elbows_on_floor and legs_raised and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and not legs_raised:
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


# Приседания
def sqats(frame, session_data: dict, timing=False):
    if "jump_started" not in session_data or "start_time" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0, "start_time": time.time()}
        )

    jump_started, repetitions_count, p_time, start_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
        session_data["start_time"]
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Колени
        right_knee = get_points(results, 26)
        left_knee = get_points(results, 25)

        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)

        # Таз немного выше коленей
        statement_upper = (round(left_hip[1], 2) + 0.02 == round(left_knee[1], 2)) or \
                          (round(right_hip[1], 2) + 0.02 == round(right_knee[1], 2))
        # Таз немного нижу коленей
        statement_lower = (round(left_hip[1], 2) - 0.02 == round(left_knee[1], 2)) or \
                          (round(right_hip[1], 2) - 0.02 == round(right_knee[1], 2))
        # Таз равен коленям
        statement_eq = (round(left_hip[1], 2) == round(left_knee[1], 2)) or \
                       (round(right_hip[1], 2) == round(right_knee[1], 2))

        # Обычные приседания подсчет повторений
        # Условие if для начала повторения
        if (statement_upper or statement_eq or statement_lower) and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and not (statement_upper or statement_eq or statement_lower):
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
            "start_time": start_time
        }
    )

    return frame, fps, repetitions_count


# Приседания
def sqats_static(frame, session_data: dict, timing=False):
    if "jump_started" not in session_data or "start_time" not in session_data or "timer" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "timer": 0, "p_time": 0, "start_time": time.time()}
        )

    jump_started, repetitions_count, p_time, start_time, timer = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
        session_data["start_time"],
        session_data["timer"]
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Колени
        right_knee = get_points(results, 26)
        left_knee = get_points(results, 25)

        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)

        # Ступни
        right_foot = get_points(results, 28)
        left_foot = get_points(results, 27)

        l_knee_hip = make_vec(left_hip, left_knee)
        r_knee_hip = make_vec(right_hip, right_knee)

        l_knee_foot = make_vec(left_foot, left_knee)
        r_knee_foot = make_vec(right_foot, right_knee)

        # Угол в коленях для обеих ног
        left_angle = count_angle(l_knee_hip, l_knee_foot)
        right_angle = count_angle(r_knee_hip, r_knee_foot)

        if is_front_facing(left_hip, right_hip):
            # Если человек стоит лицом к камере, проверяем, что колени ниже бедер
            left_state = left_hip[1] - 0.1 <= left_knee[1] <= left_hip[1] + 0.1
            right_state = right_hip[1] - 0.1 <= right_knee[1] <= right_hip[1] + 0.1
        else:
            # Если человек стоит боком к камере, вычисляем углы
            left_state = 70 <= left_angle <= 90
            right_state = 70 <= right_angle <= 90

        if left_state and right_state:
            if not jump_started:
                jump_started = True
                start_time = time.time()  # Начало приседа
            else:
                timer += time.time() - start_time  # Время выполнения приседа
                if round(timer, 0) > repetitions_count:
                    repetitions_count += 1
                start_time = time.time()
        else:
            if jump_started:
                jump_started = False

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "timer": timer,
            "p_time": p_time,
            "start_time": start_time
        }
    )

    return frame, fps, repetitions_count


# Пресс
def press(frame, session_data: dict, timing=False):
    if "jump_started" not in session_data:
        session_data.update(
            {"jump_started": False, "repetitions_count": 0, "p_time": 0}
        )

    jump_started, repetitions_count, p_time = (
        session_data["jump_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Ступни
        right_foot = get_points(results, 28)
        left_foot = get_points(results, 27)
        # Колени
        right_knee = get_points(results, 26)
        left_knee = get_points(results, 25)
        # Бедра
        right_hip = get_points(results, 24)
        left_hip = get_points(results, 23)
        # Плечи
        right_shoulder = get_points(results, 12)
        left_shoulder = get_points(results, 11)

        # Левая часть туловища, вектор от левого бедра до левого плеча
        left_side_vec = make_vec(left_shoulder, left_hip)
        # Правая часть туловища, вектор от правого бедра до правого плеча
        right_side_vec = make_vec(right_shoulder, right_hip)
        # Левая нога, вектор от левого бедра до левого колена
        left_leg_vec = make_vec(left_knee, left_hip)
        # Правая нога, вектор от правого бедра до правого колена
        right_leg_vec = make_vec(right_knee, right_hip)
        # Левая нога, вектор от левой колена до левого ступни
        left_foot_vec = make_vec(left_foot, left_knee)
        # Правая нога, вектор от правой колена до правого ступни
        right_foot_vec = make_vec(right_foot, right_knee)

        # Высчитываем угол между вектором от бедра до колена и вектором от бедра до плеча
        left_angle = count_angle(left_side_vec, left_leg_vec)
        right_angle = count_angle(right_side_vec, right_leg_vec)
        # Вычисляем угол для проверки сгинания ног
        left_foot_angle = count_angle((-1) * left_leg_vec, left_foot_vec)
        right_foot_angle = count_angle((-1) * right_leg_vec, right_foot_vec)

        statement_body = (10.0 <= left_angle <= 45.0) or (10.0 <= right_angle <= 45.0)
        statement_foot = (left_foot_angle < 40.0) or (right_foot_angle < 40.0)
        # Условие if для начала повторения
        if statement_body and statement_foot and not jump_started:
            jump_started = True

        # Условие elif для окончания повторения
        elif jump_started and not (statement_body and statement_foot):
            jump_started = False
            repetitions_count += 1

        draw_landmarks(frame, results)

    fps, p_time = show_fps(frame, p_time)

    session_data.update(
        {
            "jump_started": jump_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


def upor_lezha(frame, session_data: dict):
    if "exercise_started" not in session_data:
        session_data.update(
            {"exercise_started": False, "repetitions_count": 0, "p_time": 0}
        )

    exercise_started, repetitions_count, p_time = (
        session_data["exercise_started"],
        session_data["repetitions_count"],
        session_data["p_time"],
    )

    frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Координаты левого и правого бедра, колена и лодыжки
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Вычисление углов
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Проверка выполнения повторения
        if left_leg_angle > 160 and right_leg_angle > 160 and not exercise_started:
            exercise_started = True

        elif exercise_started and (left_leg_angle < 90 or right_leg_angle < 90):
            exercise_started = False
            repetitions_count += 1

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - p_time)
    p_time = current_time

    frame = cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    session_data.update(
        {
            "exercise_started": exercise_started,
            "repetitions_count": repetitions_count,
            "p_time": p_time,
        }
    )

    return frame, fps, repetitions_count


@router.websocket("")
async def workout_connection(websocket: WebSocket):
    global connections

    connection_id = str(uuid.uuid4())
    await websocket.accept()

    connections[connection_id] = {"websocket": websocket, "video_frames": []}
    logger.info(f"New WebSocket connection: {connection_id}")

    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            if "type" not in data_json:
                continue
            if data_json["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            elif data_json["type"] == "reset":
                connections[connection_id]["repetitions_count"] = 0
                connections[connection_id]["video_frames"] = []
                await websocket.send_json(
                    {
                        "type": "reset",
                        "connection_id": connection_id,
                    }
                )
                continue

            if data_json.get("is_resting", False):
                await websocket.send_json(
                    {
                        "type": "rest",
                        "connection_id": connection_id,
                    }
                )
                continue

            exercise_type = data_json["type"]

            if data_json.get("is_downloading", False):
                continue

            video_b64 = data_json["data"]
            video_bytes = base64.b64decode(video_b64)

            np_array = np.frombuffer(video_bytes, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            # Do not save the video now
            # connections[connection_id]["video_frames"].append(video_bytes)
            # connections[connection_id]["video_frames"].append(img)

            if exercise_type == "high_knees":
                frame, _, repetitions_count = process_high_knees(
                    img, connections[connection_id]
                )
            elif exercise_type == "jumping_jacks":
                frame, _, repetitions_count = process_jumping_jacks(
                    img, connections[connection_id]
                )
            elif exercise_type == "side_lunge":
                frame, _, repetitions_count = side_lunge(
                    img, connections[connection_id]
                )
            elif exercise_type == "side_kick":
                frame, _, repetitions_count = side_kick(
                    img, connections[connection_id]
                )
            elif exercise_type == "bycicle":
                frame, _, repetitions_count = bycicle(
                    img, connections[connection_id]
                )
            elif exercise_type == "leg_swings":
                frame, _, repetitions_count = leg_swings(
                    img, connections[connection_id]
                )
            elif exercise_type == "melnica":
                frame, _, repetitions_count = melnica(
                    img, connections[connection_id]
                )
            elif exercise_type == "leg_abduption":
                frame, _, repetitions_count = leg_abduption(
                    img, connections[connection_id]
                )
            elif exercise_type == "pushups":
                frame, _, repetitions_count = pushups(
                    img, connections[connection_id]
                )
            elif exercise_type == "move_plank":
                frame, _, repetitions_count = move_plank(
                    img, connections[connection_id]
                )
            elif exercise_type == "plank":
                frame, _, repetitions_count = plank(
                    img, connections[connection_id]
                )
            elif exercise_type == "climbers_steps":
                frame, _, repetitions_count = climbers_steps(
                    img, connections[connection_id]
                )
            elif exercise_type == "kick":
                frame, _, repetitions_count = kick(
                    img, connections[connection_id]
                )
            elif exercise_type == "standing_curls":
                frame, _, repetitions_count = standing_curls(
                    img, connections[connection_id]
                )
            elif exercise_type == "pelvic_lift":
                frame, _, repetitions_count = pelvic_lift(
                    img, connections[connection_id]
                )
            elif exercise_type == "pelvic_static":
                frame, _, repetitions_count = pelvic_static(
                    img, connections[connection_id]
                )
            elif exercise_type == "leg_raises_elbow_rest":
                frame, _, repetitions_count = leg_raises_elbow_rest(
                    img, connections[connection_id]
                )
            elif exercise_type == "sqats":
                frame, _, repetitions_count = sqats(
                    img, connections[connection_id]
                )
            elif exercise_type == "sqats_static":
                frame, _, repetitions_count = sqats_static(
                    img, connections[connection_id]
                )
            elif exercise_type == "press":
                frame, _, repetitions_count = press(
                    img, connections[connection_id]
                )
            elif exercise_type == "upor_lezha":
                frame, _, repetitions_count = upor_lezha(
                    img, connections[connection_id]
                )
            else:
                frame, _, repetitions_count = process_jumping_jacks(
                    img, connections[connection_id]
                )

            # This is no longer needed as we are sending the count only
            # _, buffer = cv2.imencode(".jpg", frame)
            # b64_img = base64.b64encode(buffer.tobytes()).decode("utf-8")
            # await websocket.send_json(
            #     {
            #         "type": "image",
            #         "data": b64_img,
            #         "connection_id": connection_id,
            #     }
            # )

            await websocket.send_json(
                {
                    "type": "count",
                    "data": repetitions_count,
                    "connection_id": connection_id,
                }
            )

    except WebSocketDisconnect:
        del connections[connection_id]
        logger.info(f"WebSocket connection closed: {connection_id}")
        print(f"WebSocket connection closed: {connection_id}")
        try:
            await websocket.close()
        except:
            pass


@router.get("/download_video")
async def download_video(connection_id: str):
    global connections

    logger.info(f"Downloading video for {connection_id}")
    connection = connections.get(connection_id)
    if not connection:
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder(
                {
                    "detail": "Connection not found",
                }
            ),
        )

    video_frames = connection["video_frames"]
    if not video_frames:
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder(
                {
                    "detail": "Video frames not found",
                }
            ),
        )

    path = generate_video(video_frames)
    if not path:
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(
                {
                    "detail": "Failed to generate video",
                }
            ),
        )

    return TempFileResponse(
        path=path,
        filename=f"video_{connection_id}.mp4",
        media_type="video/mp4",
    )


def generate_video(frame_data) -> str:
    if not frame_data:
        return None

    temp_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file_path = temp_file.name

    sample_frame = cv2.imdecode(
        np.frombuffer(frame_data[0], np.uint8), cv2.IMREAD_COLOR
    )
    height, width, _ = sample_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # For MP4 format
    video = cv2.VideoWriter(temp_file_path, fourcc, DESIRED_FPS, (width, height))

    for frame_bytes in frame_data:
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        video.write(frame)

    video.release()

    return temp_file_path
