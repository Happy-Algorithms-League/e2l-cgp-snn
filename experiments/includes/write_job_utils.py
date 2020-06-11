import hashlib
import json
import os
import shutil


def mkdirp(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def fill_jobfile_template(params, template):
    template.format(**params)
    return template


def write_json(d, fn):
    with open(f"{fn}", "w") as f:
        json.dump(d, f)


def create_jobfile(params, fn="jobfile.jdf"):
    jobfile_template = ""
    with open(params["jobfile_template"], "r") as f:
        for l in f:
            jobfile_template += l

    jobfile = jobfile_template.format(**params)
    jobfile += "\n"

    with open(os.path.join(params["outputdir"], fn), "w") as f:
        f.write(jobfile)


def md5_file(fn, blocksize=2 ** 20):
    """computes md5 hash of file"""
    h = hashlib.md5()
    with open(os.path.join(fn), "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def copy_file(fn, dest):
    shutil.copy2(fn, os.path.join(dest, fn))


def copy_files(fns, dest):
    for fn in fns:
        copy_file(fn, dest)


def submit_job(params, jobfile_fn="jobfile.jdf"):
    os.chdir(params["outputdir"])
    os.system(
        "{submit_command} {jobfile}".format(
            submit_command=params["submit_command"], jobfile=jobfile_fn
        )
    )
    os.chdir(params["workingdir"])
