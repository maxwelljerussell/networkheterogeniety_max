import time
from multiprocessing import Pool
from typing import Iterable, List
from tqdm import tqdm
from classes.netsalt_job import NetsaltJob
from manager.jobs import run_passive_job, run_lasing_job, run_trajectory_job
from manager.database.jobs_io import update_job_status
from manager.log import info, warn, err, dbg, get_network_logger

def run_jobs(
        jobs: Iterable[NetsaltJob],
        n_workers: int = 1,
        show_progress: bool = False,
) -> List[NetsaltJob]:
    jobs = list(jobs)
    if not jobs:
        info("[run_jobs] No jobs to run.")
        return []
    
    info(f"[run_jobs] Running {len(jobs)} jobs with n_workers={n_workers}, "
         f"show_progress={show_progress}")
    
    if n_workers <= 1:
        info("[run_jobs] Using sequential execution.")
        completed = [run_job(job) for job in jobs]
        info("[run_jobs] Sequential execution finished.")
        return completed
    
    info("[run_jobs] Using multiprocessing.Pool.")
    if show_progress:
        with Pool(processes=n_workers) as pool:
            completed = list(tqdm(pool.imap_unordered(run_job, jobs), total=len(jobs)))
    else:
        with Pool(processes=n_workers) as pool:
            completed = list(pool.imap_unordered(run_job, jobs))
        
    info("[run_jobs] Parallel execution finished.")
    return completed
        
def run_job(job: NetsaltJob):
    """Execute a single netSALT job (passive, lasing, or trajectory)."""

    net_log = get_network_logger(job.network_id)
    info(f"[run_job] Starting job_id={job.job_id}, type={job.job_type}, "
         f"network={job.network_id}, pattern_idx={job.pattern_idx}")
    net_log.info(
        f"Starting job {job.job_id} "
        f"(type={job.job_type}, pattern_idx={job.pattern_idx})"
    )

    set_job_status(job, "running")
    start_time = time.time()

    try:
        if job.job_type == "passive":
            result = run_passive_job(job)

        elif job.job_type == "lasing":
            result = run_lasing_job(job)

        elif job.job_type == "trajectory":
            result = run_trajectory_job(job)

        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

        job.execution_time = time.time() - start_time
        set_job_status(job, "completed", None, job.execution_time)

        info(f"[run_job] Completed job_id={job.job_id} in {job.execution_time:.2f}s")
        net_log.info(
            f"Job {job.job_id} completed in {job.execution_time:.2f}s "
            f"(type={job.job_type})"
        )

    except Exception as e:
        job.execution_time = time.time() - start_time
        set_job_status(job, "error", str(e), job.execution_time)
        
        err(f"[run_job] ERROR in job_id={job.job_id}: {e}")
        net_log.error(
            f"ERROR in job {job.job_id} after {job.execution_time:.2f}s: {e}",
        )

    return job

def set_job_status(
        job: NetsaltJob,
        status: str,
        error_message: str | None = None,
        execution_time: float | None = None,
):
    prev_status = job.status

    job.status = status
    job.error_message = error_message
    if execution_time is not None:
        job.execution_time = execution_time
    
    update_job_status(job.job_id, status, error_message, job.execution_time)

    # logging
    msg = (f"[set_job_status] job_id={job.job_id} "
           f"{prev_status} -> {status}, "
           f"exec_time={job.execution_time}, "
           f"error={error_message}")
    if status == "error":
        err(msg)
    elif status == "completed":
        info(msg)
    else:
        dbg(msg)

    net_log = get_network_logger(job.network_id)
    if status == "error":
        net_log.error(msg)
    else:
        net_log.info(msg)