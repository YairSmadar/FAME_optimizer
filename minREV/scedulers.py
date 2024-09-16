from torch.optim.lr_scheduler import MultiStepLR, LinearLR, CyclicLR, LambdaLR
from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    @abstractmethod
    def get_scheduler(self, optimizer):
        pass

    @abstractmethod
    def condition(self, epoch):
        pass


class DefaultWarmUpScheduler(BaseScheduler):
    NAME = "defaultwarmup"

    def get_scheduler(self, optimizer):
        return LinearLR(optimizer=optimizer, start_factor=0.001, end_factor=1.0,
                        total_iters=10, last_epoch=-1)

    def condition(self, epoch):
        return epoch < 10


class DefaultScheduler(BaseScheduler):
    NAME = "deafultmultistep"

    def get_scheduler(self, optimizer):
        return MultiStepLR(optimizer=optimizer, milestones=range(30, 100),
                           gamma=0.1 ** (1 / 70))

    def condition(self, epoch):
        return True


class MultiStepLRScheduler(BaseScheduler):
    NAME = "multisteplr"

    def get_scheduler(self, optimizer):
        return MultiStepLR(optimizer=optimizer, milestones=[30, 70, 90],
                           gamma=0.5)

    def condition(self, epoch):
        return True


class ExtraMultiStepLRScheduler(BaseScheduler):
    NAME = "extramultisteplr"

    def get_scheduler(self, optimizer):
        return MultiStepLR(optimizer=optimizer, milestones=range(10, 90, 5),
                           gamma=0.95)

    def condition(self, epoch):
        return True


class LinearDecayScheduler(BaseScheduler):
    NAME = "lineardecay"

    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs

    def get_scheduler(self, optimizer):
        # Linear decay function
        lmbda = lambda epoch: 1 + 9 * ((self.total_epochs - 1 - epoch) / (self.total_epochs - 1))
        return LambdaLR(optimizer, lr_lambda=lmbda)

    def condition(self, epoch):
        return True  # Always apply this scheduler.


class SchedulerFactory:
    def __init__(self, total_epochs=100):
        self.scheduler_classes = {
            DefaultWarmUpScheduler.NAME: DefaultWarmUpScheduler,
            DefaultScheduler.NAME: DefaultScheduler,
            MultiStepLRScheduler.NAME: MultiStepLRScheduler,
            ExtraMultiStepLRScheduler.NAME: ExtraMultiStepLRScheduler,
            LinearDecayScheduler.NAME: LinearDecayScheduler
        }
        self.total_epochs = total_epochs

    def get_scheduler(self, scheduler_name):
        scheduler_class = self.scheduler_classes.get(scheduler_name)
        if scheduler_class is None:
            raise KeyError(f'There is no scheduler name {scheduler_name}')

        if scheduler_name == 'lineardecay':
            return scheduler_class(self.total_epochs)

        return scheduler_class()


class SchedulerManager:
    def __init__(self, scheduler_name, total_epochs=100):
        self.schedulers = []
        self.factory = SchedulerFactory(total_epochs)
        self.scheduler_name = scheduler_name

    def add_scheduler(self, scheduler_name, optimizer):
        scheduler = self.factory.get_scheduler(scheduler_name)
        scheduler_instance = scheduler.get_scheduler(optimizer)
        self.schedulers.append((scheduler_instance, scheduler.condition))

    def set_schedulers(self, optimizer):
        if self.scheduler_name == 'default':
            self.add_scheduler(DefaultWarmUpScheduler.NAME, optimizer)
            self.add_scheduler(DefaultScheduler.NAME, optimizer)
        elif self.scheduler_name == 'multisteplr':
            self.add_scheduler(MultiStepLRScheduler.NAME, optimizer)
        elif self.scheduler_name == 'extramultisteplr':
            self.add_scheduler(ExtraMultiStepLRScheduler.NAME, optimizer)
        elif self.scheduler_name == 'lineardecay':
            self.add_scheduler(LinearDecayScheduler.NAME, optimizer)
        else:
            raise KeyError(f'There is no scheduler name {self.scheduler_name}')

    def get_scheduler_name(self):
        return self.scheduler_name

    def schedulers_step(self, epoch):
        for scheduler, cond in self.schedulers:
            if cond(epoch):
                scheduler.step()
