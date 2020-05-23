import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

from gpu_use.db.engine import make_engine

Base = declarative_base()


class Node(Base):
    __tablename__ = "nodes"

    name = sa.Column(sa.String(32), primary_key=True)
    load = sa.Column(sa.String(64))
    update_time = sa.Column(sa.DateTime())

    def __repr__(self):
        return "<Node(name={}, load={})>".format(self.name, self.load)


class SLURMJob(Base):
    __tablename__ = "slurm_jobs"

    job_id = sa.Column(sa.Integer, primary_key=True)
    is_debug_job = sa.Column(sa.Boolean)

    node = sa.orm.relationship("Node", back_populates="slurm_jobs", lazy="select")
    node_name = sa.Column(sa.String(32), sa.ForeignKey("nodes.name"))

    user = sa.Column(sa.String(32))

    def __repr__(self):
        return "<SLURMJob<(job_id={}, user={})>".format(self.job_id, self.user)


class GPU(Base):
    __tablename__ = "gpus"

    id = sa.Column(sa.Integer, primary_key=True)
    node_name = sa.Column(sa.String(32), sa.ForeignKey("nodes.name"), primary_key=True)

    node = sa.orm.relationship("Node", back_populates="gpus", lazy="select")

    slurm_job = sa.orm.relationship("SLURMJob", back_populates="gpus", lazy="select")
    slurm_job_id = sa.Column(sa.Integer, sa.ForeignKey("slurm_jobs.job_id"))

    def __repr__(self):
        return "<GPU(gpu_id={}, node={})>".format(self.id, self.node_name)


class GPUProcess(Base):
    __tablename__ = "gpu_processes"
    __table_args__ = (
        sa.ForeignKeyConstraint(["node_name", "gpu_id"], ["gpus.node_name", "gpus.id"]),
    )

    id = sa.Column(sa.Integer, primary_key=True)
    gpu_id = sa.Column(sa.Integer, primary_key=True)
    node_name = sa.Column(sa.String(32), primary_key=True)

    gpu = sa.orm.relationship("GPU", back_populates="processes", lazy="select")

    slurm_job = sa.orm.relationship(
        "SLURMJob", back_populates="processes", lazy="select"
    )
    slurm_job_id = sa.Column(sa.Integer, sa.ForeignKey("slurm_jobs.job_id"))

    user = sa.Column(sa.String(32))
    command = sa.Column(sa.String(128))

    def __repr__(self):
        return "<GPUProcess(pid={}, node={}, gpu={}, user={}, command={})>".format(
            self.id, self.node_name, self.gpu_id, self.user, self.command
        )


GPU.processes = sa.orm.relationship(
    "GPUProcess", order_by=GPUProcess.id, back_populates="gpu", lazy="select"
)

SLURMJob.gpus = sa.orm.relation(
    "GPU", order_by=GPU.id, back_populates="slurm_job", lazy="select"
)
SLURMJob.processes = sa.orm.relation(
    "GPUProcess", order_by=GPUProcess.id, back_populates="slurm_job", lazy="select"
)


Node.slurm_jobs = sa.orm.relationship(
    "SLURMJob", order_by=SLURMJob.job_id, back_populates="node", lazy="select"
)
Node.gpus = sa.orm.relationship(
    "GPU", order_by=GPU.id, back_populates="node", lazy="select"
)


Base.metadata.create_all(make_engine())
