import os
import string
import random
import pyheif
from io import BytesIO

from PIL import Image
from sqlalchemy.sql import select
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.core.database import AsyncSessionFactory

from app.models import Exercise, Workout, WorkoutExercise, User


class TempFileResponse(FileResponse):
    def __init__(self, path: str, filename: str, *args, **kwargs):
        super().__init__(path, filename=filename, *args, **kwargs)
        self.temp_file_path = path
        self.background = BackgroundTask(self.cleanup_temp_file)

    async def cleanup_temp_file(self):
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)
            print(f"Temporary file {self.temp_file_path} has been deleted.")

    async def __call__(self, scope, receive, send):
        await super().__call__(scope, receive, send)


def generate_random_password(length: int):
    charset = string.ascii_letters + string.digits
    return "".join(random.choices(charset, k=length))


async def compress_and_save_image(
    file: UploadFile, save_path: str, quality: int = 50, size: tuple = (400, 400)
):
    if file.filename.lower().endswith(".heic"):
        heif_file = pyheif.read(await file.read())  # Чтение HEIC файла
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride
        )
        save_path = save_path.replace(".heic", ".jpg")  # Меняем расширение на .jpg
    else:
        # Обрабатываем обычные изображения (JPEG, PNG и т.д.)
        image = Image.open(BytesIO(await file.read()))

    image = image.resize(size)
    image.save(save_path, quality=quality, optimize=True)


async def insert_default_data():
    db: AsyncSession = AsyncSessionFactory()
    # noinspection PyTypeChecker
    result = await db.execute(select(Exercise).where(Exercise.id == "high_knees"))
    exists = result.scalars().first()

    if not exists:
        high_knees = Exercise(
            id="high_knees",
            name="High Knees",
            video_link="подъем коленей высоко.MOV",
            description="Бег на месте с высоким поднятием коленей - это как игра в ловкие ниндзя! Отличный способ "
            "пробудить тело и дать заряд бодрости на весь день.",
            gif_link="подъем-коленей-высоко-v1ex.gif",
        )
        db.add(high_knees)

    # noinspection PyTypeChecker
    result = await db.execute(select(Exercise).where(Exercise.id == "jumping_jacks"))
    exists = result.scalars().first()

    if not exists:
        jumping_jacks = Exercise(
            id="jumping_jacks",
            name="Jumping Jacks",
            video_link="джампинг джэк.MOV",
            description="Прыжки с разведением рук и ног, как звездочка, взлетающая в небо! Это не только весело, "
            "но и отлично закачивает энергией на весь день.",
            gif_link="джампинг-джэк.gif",
        )
        db.add(jumping_jacks)

    # noinspection PyTypeChecker
    result = await db.execute(select(Exercise).where(Exercise.id == "side_lunge"))
    exists = result.scalars().first()

    if not exists:
        side_lunge = Exercise(
            id="side_lunge",
            name="Side lunges",
            video_link="боковые выпады.MOV",
            description="Боковые выпады",
            gif_link="боковые-выпады.gif",
        )
        db.add(side_lunge)

    result = await db.execute(select(Exercise).where(Exercise.id == "side_kick"))
    exists = result.scalars().first()

    if not exists:
        side_kick = Exercise(
            id="side_kick",
            name="Cross punches",
            video_link="боковые удары.MOV",
            description="Боковые удары",
            gif_link="боковые-удары.gif",
        )
        db.add(side_kick)

    result = await db.execute(select(Exercise).where(Exercise.id == "bycicle"))
    exists = result.scalars().first()

    if not exists:
        bycicle = Exercise(
            id="bycicle",
            name="Bicycle crunches",
            video_link="велосипед.MOV",
            description="Велосипед",
            gif_link="велосипед.gif",
        )
        db.add(bycicle)

    result = await db.execute(select(Exercise).where(Exercise.id == "leg_swings"))
    exists = result.scalars().first()

    if not exists:
        leg_swings = Exercise(
            id="leg_swings",
            name="Standing leg crunches",
            video_link="махи ногами со скручиванием.MOV",
            description="Махи ногами со скручиванием",
            gif_link="махи-ногами-со-скручиванием.gif",
        )
        db.add(leg_swings)

    result = await db.execute(select(Exercise).where(Exercise.id == "melnica"))
    exists = result.scalars().first()

    if not exists:
        melnica = Exercise(
            id="melnica",
            name="Windmills",
            video_link="мельница.MOV",
            description="Мельница",
            gif_link="мельница.gif",
        )
        db.add(melnica)

    result = await db.execute(select(Exercise).where(Exercise.id == "leg_abduption"))
    exists = result.scalars().first()

    if not exists:
        leg_abduption = Exercise(
            id="leg_abduption",
            name="Kickbacks",
            video_link="отведение ноги назад.MOV",
            description="Отведение ноги назад",
            gif_link="отведение-ноги-назад.gif",
        )
        db.add(leg_abduption)

    result = await db.execute(select(Exercise).where(Exercise.id == "pushups"))
    exists = result.scalars().first()

    if not exists:
        pushups = Exercise(
            id="pushups",
            name="Push ups",
            video_link="отжимания.mp4",
            description="Отжимания",
            gif_link="отжимания.gif",
        )
        db.add(pushups)

    result = await db.execute(select(Exercise).where(Exercise.id == "move_plank"))
    exists = result.scalars().first()

    if not exists:
        move_plank = Exercise(
            id="move_plank",
            name="Up/down planks",
            video_link="планка вверх-вниз.MOV",
            description="Планка вверх-вниз",
            gif_link="планка-вверх-вниз.gif",
        )
        db.add(move_plank)

    result = await db.execute(select(Exercise).where(Exercise.id == "plank"))
    exists = result.scalars().first()

    if not exists:
        plank = Exercise(
            id="plank",
            name="Elbow plank hold",
            video_link="планка стойка на локтях.MOV",
            description="Планка",
            gif_link="планка-стойка-на-локтях.gif",
        )
        db.add(plank)

    result = await db.execute(select(Exercise).where(Exercise.id == "climbers_steps"))
    exists = result.scalars().first()

    if not exists:
        climbers_steps = Exercise(
            id="climbers_steps",
            name="Mountain climbers",
            video_link="шаги скалалаза.MOV",
            description="Шаги скалалаза",
            gif_link="шаги-скалалаза.gif",
        )
        db.add(climbers_steps)

    result = await db.execute(select(Exercise).where(Exercise.id == "kick"))
    exists = result.scalars().first()

    if not exists:
        kick = Exercise(
            id="kick",
            name="Standing kicks",
            video_link="удар ногой.MOV",
            description="Удар ногой",
            gif_link="удар-ногой.gif",
        )
        db.add(kick)

    result = await db.execute(select(Exercise).where(Exercise.id == "standing_curls"))
    exists = result.scalars().first()

    if not exists:
        standing_curls = Exercise(
            id="standing_curls",
            name="Standing knee crunches",
            video_link="скручивания стоя.MOV",
            description="Скручивания стоя",
            gif_link="скручивания-стоя.gif",
        )
        db.add(standing_curls)

    result = await db.execute(select(Exercise).where(Exercise.id == "pelvic_lift"))
    exists = result.scalars().first()

    if not exists:
        pelvic_lift = Exercise(
            id="pelvic_lift",
            name="Chute bridge",
            video_link="подъем таза.MOV",
            description="Подъем таза",
            gif_link="подъем-таза.gif",
        )
        db.add(pelvic_lift)

    result = await db.execute(select(Exercise).where(Exercise.id == "pelvic_static"))
    exists = result.scalars().first()

    if not exists:
        pelvic_static = Exercise(
            id="pelvic_static",
            name="Chute bridge static",
            video_link="Статичное удержание таза.MOV",
            description="Статичное удержание таза",
            gif_link="Статичное-удержание-таза.gif",
        )
        db.add(pelvic_static)

    result = await db.execute(select(Exercise).where(Exercise.id == "leg_raises_elbow_rest"))
    exists = result.scalars().first()

    if not exists:
        leg_raises_elbow_rest = Exercise(
            id="leg_raises_elbow_rest",
            name="Leg raises",
            video_link="подъем ног с упором на локти.MOV",
            description="Подъем ног с упором на локти",
            gif_link="подъем-ног-с-упором-на-локти.gif",
        )
        db.add(leg_raises_elbow_rest)

    result = await db.execute(select(Exercise).where(Exercise.id == "sqats"))
    exists = result.scalars().first()

    if not exists:
        sqats = Exercise(
            id="sqats",
            name="Squats",
            video_link="приседания.gif",
            description="Приседания",
            gif_link="приседания.gif",
        )
        db.add(sqats)

    result = await db.execute(select(Exercise).where(Exercise.id == "sqats_static"))
    exists = result.scalars().first()

    if not exists:
        sqats_static = Exercise(
            id="sqats_static",
            name="Sqats static",
            video_link="приседания статично.MOV",
            description="Приседания статично",
            gif_link="приседания-статично.gif",
        )
        db.add(sqats_static)

    result = await db.execute(select(Exercise).where(Exercise.id == "upor_lezha"))
    exists = result.scalars().first()

    if not exists:
        upor_lezha = Exercise(
            id="upor_lezha",
            name="Lying down",
            video_link="упор+лежа+из+положения+стоя.gif",
            description="Упор лежа из положения стоя",
            gif_link="упор-лежа-из-положения-стоя.gif",
        )
        db.add(upor_lezha)

    result = await db.execute(select(Exercise).where(Exercise.id == "press"))
    exists = result.scalars().first()

    if not exists:
        press = Exercise(
            id="press",
            name="Sit ups",
            video_link="скручивания  пресс.MOV",
            description="Пресс",
            gif_link="скручивания--пресс.gif",
        )
        db.add(press)


    # if not exists:
    #     jumping_jacks = Exercise(
    #         id="custom",
    #         name="Пользовательская",
    #         video_link="",
    #         description="Индивидуальное упражнение, выполненное по вашей собственной воле. Повторы засчитываться не будут.",
    #         gif_link="",
    #     )
    #     db.add(jumping_jacks)

    await db.commit()
    await db.close()


