from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship
from sqlalchemy.orm.session import Session

from .base import Base
import pathlib

from sonicdb import audio


class File(Base):  # type: ignore # pragma: no cover
    """File object"""

    __tablename__ = "file"
    id = Column(Integer, primary_key=True)
    """int: File database ID"""

    filepath = Column(String)
    """str: filepath of file"""

    filename = Column(String)
    """str: filename of file"""

    extension = Column(String)
    """str: extension of file"""

    sample_rate = Column(Integer)
    """int: sample rate of file"""

    start = Column(DateTime)
    """datetime: start of file"""

    end = Column(DateTime)
    """datetime: end of file"""

    duration = Column(Float)
    """float: duration of file in seconds"""

    channel_number = Column(Integer)

    channel_id = Column(Integer, ForeignKey("channel.id"))
    """int: Channel database ID"""

    channel = relationship(
        "sonicdb.database.channel.Channel",
        back_populates="files",
        enable_typechecks=False,
    )
    """Channel: Channel object"""

    sensor_id = Column(Integer, ForeignKey("sensor.id"))
    """int: Sensor database ID"""

    sensor = relationship(
        "sonicdb.database.sensor.Sensor", back_populates="files", enable_typechecks=False
    )
    """Sensor: Sensor object"""

    samples = relationship(
        "sonicdb.database.sample.Sample", back_populates="file", enable_typechecks=False
    )
    """list: list of samples generated from the file"""

    __mapper_args__ = {
        "polymorphic_identity": "file",
    }

    def __repr__(self) -> str:  # pragma: no cover
        """Returns object representation"""

        return f"File {self.id}: {self.filename}"

    def get_audio(self) -> audio.Audio:  # type: ignore # pragma: no cover
        """Returns audio of file with a specified offset in seconds and duration in seconds

        :param offset: start delay in recording (in seconds), defaults to None
        :type offset: int or float, optional
        :param duration: length of audio output (in seconds), defaults to None
        :type duration: int or float, optional
        :return: audio
        :rtype: numpy.array
        """
        session = Session.object_session(self)
        directory = session.directory
        filepath = pathlib.PurePath(directory, self.filepath)

        return audio.Audio(filepath)
