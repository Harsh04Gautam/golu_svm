import torch
import os
import time

from model import Golu
from byte_token.tokenizer import ByteLevelTokenizer

from config import Config
cfg = Config()

# torch.manual_seed(42)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_device(device)
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'


tokenizer = ByteLevelTokenizer()

model = Golu()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
best_test_loss = 1_000_000.

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=0.95)

if os.path.exists('model.pt'):
    print('\nLoading existing model\n')
    checkpoint = torch.load('model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_test_loss = checkpoint['loss']


for _ in range(1500):
    scheduler.step()


model.compile()
model.print_model_info()


for epoch in range(cfg.epochs):
    print(scheduler.get_last_lr())
    epoch_start_time = time.time()
    start_time = time.time()
    running_train_loss = 0.
    print(f"\nEPOCH {epoch+1}:")
    for step, data in enumerate(tokenizer.get_train_dataloader()):
        optimizer.zero_grad()
        input, output = data
        input = input.to(device)
        output = output.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model.forward(input, output)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        if step % 10 == 9:
            last_loss = running_train_loss/10
            elapsed = str(round(time.time() - start_time, 2)) + "s"
            print(f"step: {step+1:<10} time: {elapsed:<10} train_loss: \033[1;92m{
                last_loss:<10.4f}\033[0m")
            running_train_loss = 0.
            start_time = time.time()

    with torch.no_grad():
        model.eval()
        train_loss = 0.
        running_test_loss = 0.
        val_loss = 0.
        start_time = time.time()

        for step, data in enumerate(tokenizer.get_test_dataloader()):
            input, output = data
            input = input.to(device)
            output = output.to(device)
            _, loss = model.forward(input, output)
            running_test_loss += loss.item()
        scheduler.step()
        avg_test_loss = running_test_loss/(step+1)
        elapsed = str(round(time.time() - start_time, 2)) + "s"
        print(f"train_loss: \033[1;92m{last_loss:<10.4f}\033[0m val_loss: \033[1;92m{
              avg_test_loss:<10.4f}\033[0m")
        start_time = time.time()
        model.train()

        if cfg.save_model and avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            print(f"New best loss {best_test_loss}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': best_test_loss
            }, 'model.pt')
