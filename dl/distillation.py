import torch
from tqdm import tqdm

class TeacherModel(torch.nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.conv = torch.nn.Conv2d(1, 32, 3)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(26 * 26 * 32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def predict(self, loader):
        self.eval()
        with torch.no_grad():
            predictions = []
            possiblities = []
            for X, _ in loader:
                possiblity = self.forward(X)
                possiblities += possiblity.tolist()
                predictions += torch.argmax(possiblity, dim=1).tolist()
            return predictions, possiblity
    
class StudentModel(torch.nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.conv = torch.nn.Conv2d(1, 16, 3)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(26 * 26 * 16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def predict(self, loader):
        self.eval()
        with torch.no_grad():
            predictions = []
            possiblities = []
            for X, _ in loader:
                possiblity = self.forward(X)
                possiblities += possiblity.tolist()
                predictions += torch.argmax(possiblity, dim=1).tolist()
            return predictions, possiblity
    
class Distillation:
    def __init__(self, teacher, student, temperature=3, alpha=0.7):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def fit(self, loader, learning_rate=0.01, n_epochs=10):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)

        self.student.train()
        progress_bar = tqdm(range(len(loader) * n_epochs), desc="Training progress")
        for _ in range(n_epochs):
            for X, y in loader:
                optimizer.zero_grad()
                outputs_teacher = self.teacher(X)
                outputs_student = self.student(X)
                loss = self.distill_loss(outputs_teacher, outputs_student, y)
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
        progress_bar.close()

    def distill_loss(self, y_teacher, y_student, y_true):
        soft_teacher = torch.nn.functional.softmax(y_teacher / self.temperature, dim=1)
        log_soft_student = torch.nn.functional.log_softmax(y_student / self.temperature, dim=1)
        distillation = torch.nn.functional.kl_div(log_soft_student, soft_teacher, reduction="batchmean")
        hard_loss = torch.nn.functional.cross_entropy(y_student, y_true)
        return self.alpha * distillation + (1 - self.alpha) * hard_loss
