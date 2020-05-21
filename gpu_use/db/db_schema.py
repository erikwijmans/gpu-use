import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

from gpu_use.db.engine import engine

Base = declarative_base()


class Node(Base):
    __tablename__ = "nodes"

    name = sa.Column(sa.String, primary_key=True)
    load = sa.Column(sa.String)

    def __repr__(self):
        return "<Node(name={}, load={})>".format(self.name, self.load)


class Process(Base):
    __tablename__ = "processes"

    id = sa.Column(sa.Integer, primary_key=True)

    node = sa.orm.relationship("Node", back_populates="processes")
    node_name = sa.Column(sa.String, sa.ForeignKey("nodes.name"), primary_key=True)

    gpu = sa.orm.relationship("GPU", back_populates="processes")
    gpu_id = sa.Column(sa.Integer, sa.ForeignKey("gpus.id"))

    user = sa.Column(sa.String)

    slurm_job_id = sa.Column(sa.Integer)
    command = sa.Column(sa.String)

    def __repr__(self):
        return "<Process(pid={}, node={}, gpu={}, job_id={}, user={}, command={})>".format(
            self.id,
            self.node_name,
            self.gpu_id,
            self.slurm_job_id,
            self.user,
            self.command,
        )


class GPU(Base):
    __tablename__ = "gpus"

    id = sa.Column(sa.Integer, primary_key=True)
    node_name = sa.Column(sa.String, sa.ForeignKey("nodes.name"), primary_key=True)

    node = sa.orm.relationship("Node", back_populates="gpus")
    processes = sa.orm.relationship(
        "Process", order_by=Process.id, back_populates="gpu"
    )

    def __repr__(self):
        return "<GPU(gpu_id={}, node={})>".format(self.id, self.node_name)


Node.gpus = sa.orm.relationship("GPU", order_by=GPU.id, back_populates="node")
Node.processes = sa.orm.relationship(
    "Process", order_by=Process.id, back_populates="node"
)


Base.metadata.create_all(engine)
