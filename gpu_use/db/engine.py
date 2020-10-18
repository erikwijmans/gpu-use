import json

from sqlalchemy import create_engine


def make_engine():
    with open("/usr/local/gpu-use/gpu-use-engine-secrets.json", "rt") as f:
        engine_secrets = json.load(f)

    engine = create_engine(
        "mysql://{}:{}@{}/gpu_use_db".format(
            engine_secrets["user"],
            engine_secrets["password"],
            engine_secrets["hostname"],
        ),
        echo=False,
        pool_size=15,
        max_overflow=30,
    )
    #  engine = create_engine("sqlite:///:memory:", echo=False)

    return engine


engine = make_engine()
