from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship

from .base import Base


class Event(Base):  # type: ignore
    __tablename__ = "event"
    id = Column(Integer, primary_key=True)
    """int: Run database ID"""

    name = Column(String)
    """str: Run name"""

    start = Column(DateTime)
    """datetime.Datetime: Run start time"""

    end = Column(DateTime)
    """datetime.Datetime: Run end time"""

    description = Column(String)

    subject_id = Column(Integer, ForeignKey("subject.id"))
    subject = relationship(
        "sonicdb.database.subject.Subject",
        back_populates="events",
        enable_typechecks=False,
    )
    """list: list of Subjects featured in run"""

    samples = relationship(
        "sonicdb.database.sample.Sample", back_populates="event", enable_typechecks=False
    )
    """list: list of Samples featured in run"""

    channels = relationship(
        "sonicdb.database.event.EventChannel",
        back_populates="event",
        enable_typechecks=False,
    )

    __mapper_args__ = {
        "polymorphic_identity": "event",
    }

    def __repr__(self) -> str:  # pragma: no cover
        """Returns object representation"""

        return f"Event: {self.name}"


class EventChannel(Base):  # type: ignore
    """A run might have multiple channels. This ties a run object to channel object through a logger id."""

    __tablename__ = "event_channel"

    event_id = Column(ForeignKey("event.id"), primary_key=True)  # type: ignore
    """int: Run database ID"""
    event = relationship(
        "sonicdb.database.event.Event", back_populates="channels", enable_typechecks=False
    )
    """Run: Run object"""
    channel_id = Column(ForeignKey("channel.id"), primary_key=True)  # type: ignore
    """int: Channel database ID"""
    channel = relationship(
        "sonicdb.database.channel.Channel",
        back_populates="events",
        enable_typechecks=False,
    )
    """Channel: Channel object"""

    __mapper_args__ = {
        "polymorphic_identity": "event_channel",
    }

    def __repr__(self) -> str:  # pragma: no cover
        """Returns object representation"""
        return f"{self.event} {self.channel}"
