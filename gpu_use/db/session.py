from sqlalchemy.orm import Session, sessionmaker

from gpu_use.db.engine import make_engine


def SessionMaker() -> Session:
    return Session(bind=make_engine())
