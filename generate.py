import torch
import os

from model import Golu
from byte_token.tokenizer import ByteLevelTokenizer

from config import Config
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_device(device)
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
cfg = Config()

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'


tokenizer = ByteLevelTokenizer()

model = Golu()

if os.path.exists('model.pt'):
    print('\nLoading existing model\n')
    checkpoint = torch.load('model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])


model.compile()
model.print_model_info()

total = 0
correct = 0

with torch.no_grad():
    model.eval()
    train_loss = 0.
    running_test_loss = 0.
    val_loss = 0.

    for step, data in enumerate(tokenizer.get_test_dataloader()):
        input, output = data
        input = input.to(device)
        output = output.to(device)
        logits, _ = model.forward(input)
        _, predicted = torch.max(logits, 1)
        total += logits.size(0)
        correct += (predicted == output).sum().item()
    accuracy = 100 * correct/total
    model.train()

print(f"Validation Accuracy: {accuracy:.2f}%")
