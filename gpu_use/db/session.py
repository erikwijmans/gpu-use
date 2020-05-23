from sqlalchemy.orm import sessionmaker

from gpu_use.db.engine import engine

SessionMaker = sessionmaker(bind=engine)
