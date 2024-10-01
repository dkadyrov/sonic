# type: ignore

import pathlib

import librosa
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists

from sonic import audio
from sonic.models import Base, Channel, Sensor, Event, Subject, File
from sonic import utilities

from datetime import datetime


class Database:  # pragma: no cover
    """SONIC Database class"""

    def __init__(self, db: str):
        # TODO Add support for other databases

        # if classification is True:
        #     from sonic.database.classification import Classification

        self.engine = create_engine(f"sqlite:///{db}")
        if database_exists(self.engine.url):
            Base.metadata.bind = self.engine
        else:
            Base.metadata.create_all(self.engine)
        DBSession = sessionmaker(bind=self.engine, autoflush=False)

        self.session = DBSession()
        """
        Inherits the DBSession class from SQLAlchemy. `Available here <https://docs.sqlalchemy.org/en/14/orm/session.html>`_.
        """

        # TODO Set values through JSON and XML input
        self.session.sample_duration = 60
        """int: duration of the sample in seconds"""

        self.session.sample_overlap = 0
        """int: overlap of the sample in seconds"""

        self.session.sample_rate = 25000
        """int: default sample rate """

        self.session.directory = pathlib.Path(db).parent

    def get_audio(
        self,
        start: datetime = None,
        end: datetime = None,
        event: Event = None,
        sensor: Sensor = None,
        channel: Channel = None,
        channel_number: int = None,
    ) -> audio.Audio:  #
        init_start = start
        init_end = end

        if start:
            if not isinstance(start, utilities.datetime):
                start = utilities.read_datetime(start)

        if end:
            if not isinstance(start, utilities.datetime):
                end = utilities.read_datetime(end)

        if event:
            if not start:
                start = event.start
            if not end:
                end = event.end

        if channel:
            if isinstance(channel, Channel):
                channel = channel
                sensor = channel.sensor
                channel_number = channel.number

        elif sensor:
            if isinstance(sensor, Sensor):
                sensor = sensor
            else:
                sensor = self.get_sensor(sensor)

        files = (
            self.session.query(File)
            .filter(File.start <= end)
            .filter(File.end >= start)
            .filter(File.sensor == sensor)
            .filter(File.channel_number == channel_number)
            .all()
        )

        if len(files) == 0:
            return None

        sample_rate = files[0].sample_rate

        file_start = start

        length = abs((end - start).total_seconds() * sample_rate)

        data = []
        for file in files:
            file.filepath = pathlib.PurePath(self.session.directory, file.filepath)
            offset = (start - file.start).total_seconds()

            if offset < 0:
                data.extend([0] * int(-offset * sample_rate))
                offset = 0

            duration = (file.end - end).total_seconds()

            if duration < 0:
                if offset >= file.duration:
                    data.extend(librosa.load(file.filepath, sr=sample_rate)[0].tolist())
                else:
                    data.extend(
                        librosa.load(file.filepath, offset=offset, sr=sample_rate)[
                            0
                        ].tolist()
                    )
            else:
                data.extend(
                    librosa.load(
                        file.filepath,
                        offset=offset,
                        duration=file.duration - duration - offset,
                        sr=sample_rate,
                    )[0].tolist()
                )

            start = file.end

        data.extend([0.0] * int(length - len(data)))

        a = audio.Audio(
            audio=np.asarray(data), sample_rate=sample_rate, start=file_start
        )
        a = a.trim(init_start, init_end)

        return a

    def get_sensor(
        self, sensor: Sensor | int | dict[str, int] | str
    ) -> Sensor | None:  #
        if isinstance(sensor, Sensor):
            return sensor
        elif isinstance(sensor, int):
            return self.session.query(Sensor).get(sensor)
        elif isinstance(sensor, dict):
            sensor = utilities.lower_keys(sensor)
            s = (
                self.session.query(Sensor)
                .filter(
                    Sensor.name == sensor["name"],
                    Sensor.subname == sensor["subname"],
                )
                .all()
            )
            if len(s) == 0:
                return None
            return s[0]
        elif isinstance(sensor, str):
            s = self.session.query(Sensor).filter(Sensor.name == sensor).all()
            if len(s) == 0:
                return None
            return s[0]

        return None

    def get_subject(
        self, subject: Subject | int | dict[str, int] | str
    ) -> Subject | None:  #
        if isinstance(subject, Subject):
            return subject
        elif isinstance(subject, int):
            return self.session.query(Subject).get(subject)
        elif isinstance(subject, dict):
            subject = utilities.lower_keys(subject)
            s = (
                self.session.query(Subject)
                .filter(Subject.name == subject["name"])
                .all()
            )
            if len(s) == 0:
                return None
            return s[0]
        elif isinstance(subject, str):
            s = self.session.query(Subject).filter(Subject.name == subject).all()
            if len(s) == 0:
                return None
            return s[0]

        return None

    def get_channel(
        self, channel: Channel | int | dict[str, int], sensor: Sensor | None = None
    ) -> Channel | None:  #
        if isinstance(channel, Channel):
            return channel
        elif isinstance(channel, int):
            if sensor is None:
                return self.session.query(Channel).get(channel)
            else:
                channel = {"number": channel}

        channel = utilities.lower_keys(channel)
        if sensor:
            channel["sensor"] = sensor

        if not isinstance(channel["sensor"], Sensor):
            channel["sensor"] = self.get_sensor(channel["sensor"])

        c = (
            self.session.query(Channel)
            .filter(Channel.number == channel["number"])
            .filter(Channel.sensor == channel["sensor"])
            .all()
        )

        if len(c) == 0:
            return None

        return c[0]

    # TODO Add sample support

    # TODO Add resample support
