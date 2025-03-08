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
        return super().store_parts() + (
            f"mx_{self.mx}",
            f"my_{self.my}",
        )


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
    s_ratio_index = luigi.IntParameter(default=5)

    @property
    def s_ratio(self):
        conversion = {
            0: 0.0,
            1: 0.00031622776601683794,
            2: 0.0005551935914386209,
            3: 0.0009747402255566064,
            4: 0.001711328304161781,
            5: 0.0030045385302046933,
            6: 0.00527499706370262,
            7: 0.009261187281287938,
            8: 0.01625964693881482,
            9: 0.02854667663497933,
            10: 0.05011872336272722,
        }

        return conversion[self.s_ratio_index]

    def store_parts(self):
        return super().store_parts() + (
            f"s_index_{self.s_ratio_index}_ratio_{str_encode_value(self.s_ratio)}",
        )


class TemplateRandomMixin:

    train_random_seed = luigi.IntParameter(default=233)

    def store_parts(self):
        return super().store_parts() + (f"train_seed_{self.train_random_seed}",)


class TranvalSplitRandomMixin:

    trainval_split_seed = luigi.IntParameter(default=0)

    def store_parts(self):
        return super().store_parts() + (
            f"trainval_split_seed_{self.trainval_split_seed}",
        )


class TestSetMixin:

    use_true_mu = luigi.BoolParameter(default=True)
    test_set_fold = luigi.IntParameter(default=0)

    def store_parts(self):
        return super().store_parts() + (
            f"use_true_mu_{self.use_true_mu}",
            f"test_set_fold_{self.test_set_fold}",
        )


class BkgTemplateUncertaintyMixin:

    num_bkg_templates = luigi.IntParameter(default=5)

    def store_parts(self):
        return super().store_parts() + (f"num_templates_{self.num_bkg_templates}",)


class BkgModelMixin:

    use_perfect_bkg_model = luigi.BoolParameter(default=False)

    def store_parts(self):
        return super().store_parts() + (
            f"use_perfect_bkg_model_{self.use_perfect_bkg_model}",
        )


class SigTemplateTrainingUncertaintyMixin:

    # controls the random seed for the training
    train_num_sig_templates = luigi.IntParameter(default=5)

    def store_parts(self):
        return super().store_parts() + (
            f"train_num_templates_{self.train_num_sig_templates}",
        )


class TranvalSplitUncertaintyMixin:

    # controls the random seed for the splitting train val set
    split_num_sig_templates = luigi.IntParameter(default=5)

    def store_parts(self):
        return super().store_parts() + (
            f"split_num_templates_{self.split_num_sig_templates}",
        )


class WScanMixin:

    w_min = luigi.FloatParameter(default=0.00001)
    w_max = luigi.FloatParameter(default=0.05)
    scan_number = luigi.IntParameter(default=20)

    def store_parts(self):
        return super().store_parts() + (
            f"w_min_{self.w_min}_w_max_{self.w_max}_scan_{self.scan_number}",
        )
