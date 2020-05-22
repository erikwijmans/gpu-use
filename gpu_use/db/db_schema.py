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


class SLURMJob(Base):
    __tablename__ = "slurm_jobs"

    job_id = sa.Column(sa.String, primary_key=True)

    node = sa.orm.relationship("Node", back_populates="slurm_jobs", lazy="joined")
    node_name = sa.Column(sa.String, sa.ForeignKey("nodes.name"))

    user = sa.Column(sa.String)

    def __repr__(self):
        return "<SLURMJob<(job_id={}, user={})>".format(self.job_id, self.user)


class Process(Base):
    __tablename__ = "processes"

    id = sa.Column(sa.Integer, primary_key=True)

    node = sa.orm.relationship("Node", back_populates="processes", lazy="joined")
    node_name = sa.Column(sa.String, sa.ForeignKey("nodes.name"), primary_key=True)

    gpu = sa.orm.relationship("GPU", back_populates="processes", lazy="joined")
    gpu_id = sa.Column(sa.Integer, sa.ForeignKey("gpus.id"))

    slurm_job = sa.orm.relationship(
        "SLURMJob", back_populates="processes", lazy="joined"
    )
    slurm_job_id = sa.Column(sa.String, sa.ForeignKey("slurm_jobs.job_id"))

    user = sa.Column(sa.String)
    command = sa.Column(sa.String)

    def __repr__(self):
        return "<Process(pid={}, node={}, gpu={}, user={}, command={})>".format(
            self.id, self.node_name, self.gpu_id, self.user, self.command
        )


class GPU(Base):
    __tablename__ = "gpus"

    id = sa.Column(sa.Integer, primary_key=True)
    node_name = sa.Column(sa.String, sa.ForeignKey("nodes.name"), primary_key=True)

    node = sa.orm.relationship("Node", back_populates="gpus", lazy="joined")
    processes = sa.orm.relationship(
        "Process", order_by=Process.id, back_populates="gpu", lazy="joined"
    )

    slurm_job = sa.orm.relationship("SLURMJob", back_populates="gpus", lazy="joined")
    slurm_job_id = sa.Column(sa.String, sa.ForeignKey("slurm_jobs.job_id"))
    slurm_job_user = sa.Column(sa.String, sa.ForeignKey("slurm_jobs.user"))

    def __repr__(self):
        return "<GPU(gpu_id={}, node={})>".format(self.id, self.node_name)


SLURMJob.gpus = sa.orm.relation(
    "GPU", order_by=GPU.id, back_populates="slurm_job", lazy="joined"
)
SLURMJob.processes = sa.orm.relation(
    "Process", order_by=Process.id, back_populates="slurm_job", lazy="joined"
)

Node.slurm_jobs = sa.orm.relationship(
    "SLURMJob", order_by=SLURMJob.job_id, back_populates="node", lazy="joined"
)
Node.gpus = sa.orm.relationship(
    "GPU", order_by=GPU.id, back_populates="node", lazy="joined"
)
Node.processes = sa.orm.relationship(
    "Process", order_by=Process.id, back_populates="node", lazy="joined"
)


Base.metadata.create_all(engine)
