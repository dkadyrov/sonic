from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship

from .base import Base


class Sample(Base):  # type: ignore
    """Sample object"""

    __tablename__ = "sample"
    id = Column(Integer, primary_key=True)
    """int: Sample database ID"""

    filepath = Column(String)
    """str: Filepath of sample"""

    event_id = Column(Integer, ForeignKey("event.id"))
    """int: Run database ID"""

    event = relationship(
        "sonic.database.event.Event", back_populates="samples", enable_typechecks=False
    )
    """Run: Run object"""

    sensor_id = Column(Integer, ForeignKey("sensor.id"))
    """int: Sensor database ID"""

    sensor = relationship(
        "sonic.database.sensor.Sensor",
        back_populates="samples",
        enable_typechecks=False,
    )
    """Sensor: Sensor object"""

    channel_id = Column(Integer, ForeignKey("channel.id"))
    """int: Channel database ID"""

    channel = relationship(
        "sonic.database.channel.Channel",
        back_populates="samples",
        enable_typechecks=False,
    )
    """Channel: Channel object"""

    subject_id = Column(Integer, ForeignKey("subject.id"))
    """int: Subject database ID"""

    subject = relationship(
        "sonic.database.subject.Subject",
        back_populates="samples",
        enable_typechecks=False,
    )
    """Subject: Subject object"""

    datetime = Column(DateTime)
    """datetime.Datetime: datetime of sample data"""

    file_id = Column(Integer, ForeignKey("file.id"))
    """int: File database ID where sample audio comes from"""

    file = relationship(
        "sonic.database.file.File", back_populates="samples", enable_typechecks=False
    )
    """File: File object where sample audio comes from"""

    classifications = relationship(
        "sonic.database.classification.Classification",
        back_populates="sample",
        enable_typechecks=False,
    )
    """list: list classifications generated from sample"""

    __mapper_args__ = {
        "polymorphic_identity": "sample",
    }
