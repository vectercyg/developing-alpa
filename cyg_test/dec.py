from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph.config import Config

class Banana:

    def eat(self):
        pass


class Person:

    def __init__(self):
        self.no_bananas()

    def no_bananas(self):
        self.bananas = []

    def add_banana(self, banana):
        self.bananas.append(banana)

    def eat_bananas(self):
        [banana.eat() for banana in self.bananas]
        self.no_bananas()


def main():
    dot_file_path="cyg_test/pycallgraph.dot"
    full_func_name_file="/home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/cyg_test/full_func_name_file.txt"
    tracker_log="/home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/cyg_test/trace.dot"
    with PyCallGraph(output=GraphvizOutput(dot_file_path=dot_file_path), config=Config(verbose=True,tracker_log=tracker_log)):
        person = Person()
        for a in range(10):
            person.add_banana(Banana())
        person.eat_bananas()
    pass


if __name__ == '__main__':
    main()
    pass