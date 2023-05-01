from tensorboardX import SummaryWriter


def get_summary_writer(logdir=None, comment='', purge_step=None, max_queue=10,
                       flush_secs=120, filename_suffix='', write_to_disk=True, log_dir=None, **kwargs):
    tb = SummaryWriter(logdir, comment, purge_step, max_queue, flush_secs, filename_suffix,
                       write_to_disk, log_dir, **kwargs)
    return tb
