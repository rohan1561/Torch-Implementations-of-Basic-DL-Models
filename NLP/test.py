from absl import app, flags, logging
import sh
import torch
import pytorch_lightning as pl
import nlp
import transformers

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_float('momentum', 0.9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('batch_size', 8, '')
FLAGS = flags.FLAGS
sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

class Sentiment(pl.LightningModule):
    def __init__(self):
        super(Sentiment, self).__init__()
        self.model = transformers.BertForSequenceClassification.\
                from_pretrained(FLAGS.model)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)

        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(
                    x['text'],
                    max_length=FLAGS.seq_length,
                    pad_to_max_length=True,
                    )
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset('imdb', 
                split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else "5%"}]')
            ds = ds.map(_tokenize)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        #acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        #losses = torch.cat(outputs, 0).mean()
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def train_dataloader(self):
        print('making batches', 'xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        return torch.utils.data.DataLoader(
                self.train_ds,
                batch_size = FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                self.test_ds,
                batch_size = FLAGS.batch_size,
                drop_last=False,
                shuffle=False,
                )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                lr=FLAGS.lr, momentum=FLAGS.momentum,
                )
        return optimizer

def main(_):
    model = Sentiment()
    trainer = pl.Trainer(default_root_dir='logs',
            gpus=(1 if torch.cuda.is_available() else 0),
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            logger=pl.loggers.TensorBoardLogger('logs/', name='imdb',
                version=0),
            )

    trainer.fit(model)

if __name__ == '__main__':
    app.run(main)

