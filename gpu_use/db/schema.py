import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

from gpu_use.db.engine import engine

Base = declarative_base()


class Lab(Base):
    __tablename__ = "labs"

    name = sa.Column(sa.String(32), primary_key=True)

    def __repr__(self):
        return "<Lab(name={})>".format(self.name)


user_node_association_table = sa.Table(
    "user_node_association_table",
    Base.metadata,
    sa.Column("user_name", sa.String(32), sa.ForeignKey("users.name")),
    sa.Column("node_name", sa.String(32), sa.ForeignKey("nodes.name")),
)


class User(Base):
    __tablename__ = "users"

    name = sa.Column(sa.String(32), primary_key=True)

    lab = sa.orm.relationship("Lab", back_populates="users", lazy="select")
    lab_name = sa.Column(sa.String(32), sa.ForeignKey("labs.name"))

    nodes = sa.orm.relationship(
        "Node",
        secondary=user_node_association_table,
        back_populates="users",
        lazy="select",
    )

    def __repr__(self):
        return "<User(name={}, lab_name={})>".format(self.name, self.lab_name)


class Node(Base):
    __tablename__ = "nodes"

    name = sa.Column(sa.String(32), primary_key=True)
    load = sa.Column(sa.String(64))
    update_time = sa.Column(sa.DateTime())

    users = sa.orm.relationship(
        "User",
        secondary=user_node_association_table,
        back_populates="nodes",
        lazy="select",
    )

    def __repr__(self):
        return "<Node(name={}, load={})>".format(self.name, self.load)


class SLURMJob(Base):
    __tablename__ = "slurm_jobs"

    job_id = sa.Column(sa.Integer, primary_key=True)
    is_debug_job = sa.Column(sa.Boolean)
    cpus = sa.Column(sa.Integer)

    node = sa.orm.relationship("Node", back_populates="slurm_jobs", lazy="select")
    node_name = sa.Column(sa.String(32), sa.ForeignKey("nodes.name"))

    lab = sa.orm.relationship("Lab", back_populates="slurm_jobs", lazy="select")
    lab_name = sa.Column(sa.String(32), sa.ForeignKey("labs.name"))

    user = sa.orm.relationship("User", back_populates="slurm_jobs", lazy="select")
    user_name = sa.Column(sa.String(32), sa.ForeignKey("users.name"))

    def __repr__(self):
        return "<SLURMJob<(job_id={}, user={})>".format(self.job_id, self.user)


class GPU(Base):
    __tablename__ = "gpus"

    id = sa.Column(sa.Integer, primary_key=True)
    node_name = sa.Column(sa.String(32), sa.ForeignKey("nodes.name"), primary_key=True)

    node = sa.orm.relationship("Node", back_populates="gpus", lazy="select")

    slurm_job = sa.orm.relationship("SLURMJob", back_populates="gpus", lazy="select")
    slurm_job_id = sa.Column(sa.Integer, sa.ForeignKey("slurm_jobs.job_id"))

    lab = sa.orm.relationship("Lab", back_populates="gpus", lazy="select")
    lab_name = sa.Column(sa.String(32), sa.ForeignKey("labs.name"))

    user = sa.orm.relationship("User", back_populates="gpus", lazy="select")
    user_name = sa.Column(sa.String(32), sa.ForeignKey("users.name"))

    update_time = sa.Column(sa.DateTime())

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

    user = sa.orm.relationship("User", back_populates="processes", lazy="select")
    user_name = sa.Column(sa.String(32), sa.ForeignKey("users.name"))

    command = sa.Column(sa.String(128))

    def __repr__(self):
        return "<GPUProcess(pid={}, node={}, gpu={}, user={}, command={})>".format(
            self.id, self.node_name, self.gpu_id, self.user, self.command
        )


Lab.users = sa.orm.relationship(
    "User", order_by=User.name, back_populates="lab", lazy="select"
)

Lab.slurm_jobs = sa.orm.relationship(
    "SLURMJob", order_by=SLURMJob.job_id, back_populates="lab", lazy="select"
)

Lab.gpus = sa.orm.relationship(
    "GPU", order_by=GPU.id, back_populates="lab", lazy="select"
)


User.slurm_jobs = sa.orm.relationship(
    "SLURMJob", order_by=SLURMJob.job_id, back_populates="user", lazy="select"
)

User.gpus = sa.orm.relationship(
    "GPU", order_by=GPU.id, back_populates="user", lazy="select"
)

User.processes = sa.orm.relation(
    "GPUProcess", order_by=GPUProcess.id, back_populates="user", lazy="select"
)

Node.gpus = sa.orm.relationship(
    "GPU", order_by=GPU.id, back_populates="node", lazy="select"
)

Node.slurm_jobs = sa.orm.relationship(
    "SLURMJob", order_by=SLURMJob.job_id, back_populates="node", lazy="select"
)


SLURMJob.gpus = sa.orm.relation(
    "GPU", order_by=GPU.id, back_populates="slurm_job", lazy="select"
)

SLURMJob.processes = sa.orm.relation(
    "GPUProcess", order_by=GPUProcess.id, back_populates="slurm_job", lazy="select"
)


GPU.processes = sa.orm.relationship(
    "GPUProcess", order_by=GPUProcess.id, back_populates="gpu", lazy="select"
)


Base.metadata.create_all(engine)
