from torchvision import models
import torch 
class Event_record_custom():
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    def record_start(self):
        self.start.record(check_graph_time=torch.cuda.is_current_stream_capturing())
    def record_end(self):
        self.end.record(check_graph_time=torch.cuda.is_current_stream_capturing())
    def __enter__(self):
        self.record_start()
    def __exit__(self, exc_type, exc_value, traceback):
        self.record_end()
    def get_time(self):
        return self.start.elapsed_time(self.end)
    
    
class Event_record():
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    def record_start(self):
        self.start.record()
    def record_end(self):
        self.end.record()
    def __enter__(self):
        self.record_start()
    def __exit__(self, exc_type, exc_value, traceback):
        self.record_end()
    def get_time(self):
        return self.start.elapsed_time(self.end)

record = Event_record()


model = models.densenet121().cuda()


batch_size = 1
input = torch.randn(batch_size, 3, 224, 224).cuda()


with record:
    output = model(input)

torch.cuda.synchronize()
print(record.get_time())
