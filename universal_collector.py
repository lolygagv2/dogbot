# universal_collector.py
class UniversalDataCollector:
    def __init__(self):
        self.camera = Picamera2()
        self.classes = []
        
    def add_class(self, name, category):
        """Add any class - dog or food"""
        self.classes.append({
            'name': name,
            'category': category,  # 'dog' or 'food'
            'samples': 0
        })
    
    def collect_samples(self):
        for cls in self.classes:
            print(f"Collecting {cls['name']}...")
            # Same collection logic for everything