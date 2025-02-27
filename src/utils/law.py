import os
import subprocess

import luigi
import law
import pandas as pd

from src.utils.utils import str_encode_value

class BaseTask(law.Task):
    """
    Base task which provides some convenience methods
    """

    version = law.Parameter()

    def store_parts(self):
        task_name = self.__class__.__name__
        return (
            os.getenv("OUTPUT_DIR"),
            f"version_{self.version}",
            task_name,
        )

    def local_path(self, *path):
        sp = self.store_parts()
        sp += path
        return os.path.join(*(str(p) for p in sp))

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)

    def local_directory_target(self, *path, **kwargs):
        return law.LocalDirectoryTarget(self.local_path(*path), **kwargs)


class ProcessMixin:

    mx = luigi.IntParameter(default=100)
    my = luigi.IntParameter(default=500)

    def store_parts(self):
        return super().store_parts() + (f"mx_{self.mx}", f"my_{self.my}",)


class BkginSRDataMixin:
    """
    In order to compare with PAWS, we have to use the same number of bkgs
    in SR to make our fake data. This variavle defines this number which is 
    num of bkgs in our data. 50% will go into training, 25% into validation and
    25% into testing
    """
    bkg_num_in_sr_data = luigi.IntParameter(default=366570)


class SignalStrengthMixin:

    # S/(S+B) ratio
    s_ratio = luigi.FloatParameter(default=0.005)

    def store_parts(self):
        return super().store_parts() + (f"s_ratio_{str_encode_value(self.s_ratio)}",)


class TemplateRandomMixin:

    train_random_seed = luigi.IntParameter(default=233)

    def store_parts(self):
        return super().store_parts() + (f"train_seed_{self.train_random_seed}",)


class TranvalSplitRandomMixin:

    sample_random_seed = luigi.IntParameter(default=42)

    def store_parts(self):
        return super().store_parts() + (f"trainval_split_seed_{self.sample_random_seed}",)

    
class BkgTemplateUncertaintyMixin:

    num_bkg_templates = luigi.IntParameter(default=5)

    def store_parts(self):
        return super().store_parts() + (f"num_templates_{self.num_bkg_templates}",)
    
class SigTemplateTrainingUncertaintyMixin:

    # controls the random seed for the training
    train_num_sig_templates = luigi.IntParameter(default=5)

    def store_parts(self):
        return super().store_parts() + (f"train_num_templates_{self.train_num_sig_templates}",)

class SigTemplateSplittingUncertaintyMixin:
    
    # controls the random seed for the splitting train val set
    split_num_sig_templates = luigi.IntParameter(default=10)

    def store_parts(self):
        return super().store_parts() + (f"split_num_templates_{self.split_num_sig_templates}",)