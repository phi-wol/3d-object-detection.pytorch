from tqdm import tqdm
from dataclasses import dataclass

from torchdet3d.evaluation import compute_average_distance, compute_accuracy
from torchdet3d.utils import AverageMeter, save_snap


@dataclass
class Trainer:
    model: object
    train_loader: object
    optimizer: object
    criterions: list
    writer: object
    max_epoch : float
    log_path : str
    device : str ='cuda:0'
    save_chkpt: bool = True
    debug: bool = False
    train_step: int = 0

    def train(self, epoch):
        ''' procedure launching main training'''

        losses = AverageMeter()
        ADD_meter = AverageMeter()
        SADD_meter = AverageMeter()
        ACC_meter = AverageMeter()

        # switch to train mode and train one epoch
        self.model.train()
        reg_criterion, class_criterion = self.criterions
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for it, (imgs, gt_kp, gt_cats) in loop:
            # put image and keypoints on the appropriate device
            imgs, gt_kp, gt_cats = self.put_on_device([imgs, gt_kp, gt_cats], self.device)
            # compute output and loss
            pred_kp, pred_cats = self.model(imgs, gt_cats)
            reg_loss = reg_criterion(pred_kp, gt_kp)
            if class_criterion is not None:
                class_loss = class_criterion(pred_cats, gt_cats)
                loss = class_loss + reg_loss # FIX losses contribution
            else:
                loss = reg_loss
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # measure metrics
            ADD, SADD = compute_average_distance(pred_kp, gt_kp)
            acc = compute_accuracy(pred_cats, gt_cats)
            # record loss
            losses.update(loss.item(), imgs.size(0))
            ADD_meter.update(ADD, imgs.size(0))
            SADD_meter.update(SADD, imgs.size(0))
            ACC_meter.update(acc, imgs.size(0))
            # write to writer for tensorboard
            self.writer.add_scalar('Train/loss', loss.item(), global_step=self.train_step)
            self.writer.add_scalar('Train/ADD', ADD_meter.avg, global_step=self.train_step)
            self.writer.add_scalar('Train/SADD', SADD_meter.avg, global_step=self.train_step)
            self.writer.add_scalar('Train/ACC', ACC_meter.avg, global_step=self.train_step)
            self.train_step += 1
            # update progress bar
            loop.set_description(f'Epoch [{epoch}/{self.max_epoch}]')
            loop.set_postfix(loss=loss, avr_loss = losses.avg,
                             ADD=ADD, avr_ADD=ADD_meter.avg, SADD=SADD,
                             avr_SADD=SADD_meter.avg, acc=acc, acc_avg = ACC_meter.avg,
                             lr=self.optimizer.param_groups[0]['lr'])

            if self.debug and it == 10:
                break

        if self.save_chkpt:
            save_snap(self.model, self.optimizer, epoch, self.log_path)

        print(f"train: epoch: {epoch}, ADD: {ADD_meter.avg},"
              f" SADD: {SADD_meter.avg}, loss: {losses.avg}, accuracy: {ACC_meter.avg}")

    @staticmethod
    def put_on_device(items, device):
        for i in range(len(items)):
            items[i] = items[i].to(device)
        return items
