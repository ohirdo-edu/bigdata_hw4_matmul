from functools import reduce
import operator

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import TextValueProtocol


class MatMul(MRJob):
    OUTPUT_PROTOCOL = TextValueProtocol

    def configure_args(self):
        super().configure_args()

        self.add_passthru_arg('--first_dim')
        self.add_passthru_arg('--mid_dim')
        self.add_passthru_arg('--last_dim')

    def mult_mapper_init(self):
        self.first_dim = int(self.options.first_dim)
        self.mid_dim = int(self.options.mid_dim)
        self.last_dim = int(self.options.last_dim)

    def mult_mapper(self, _, line):
        label, raw_row, *raw_row_values = line.split()
        row = int(raw_row)
        for col, raw_value in enumerate(raw_row_values):
            value = float(raw_value)

            if label == 'a':
                for k in range(self.last_dim):
                    yield (row, k, col), value
            if label == 'b':
                for k in range(self.first_dim):
                    yield (k, col, row), value

    def mult_reducer(self, key, values):
        yield key, reduce(operator.mul, values, 1)

    def sum_mapper(self, key, value):
        row, col, index = key
        yield (row, col), value

    def sum_reducer(self, key, values):
        yield key, sum(values)

    def out_mapper(self, key, value):
        row, col = key
        yield row, (col, value)

    def out_reducer(self, key, values):
        yield key, f"out {key} {' '.join(str(value) for _, value in sorted(values))}"

    def steps(self):
        return [
            MRStep(
                mapper_init=self.mult_mapper_init,
                mapper=self.mult_mapper,
                reducer=self.mult_reducer,
            ),
            MRStep(
                mapper=self.sum_mapper,
                reducer=self.sum_reducer,
            ),
            MRStep(
                mapper=self.out_mapper,
                reducer=self.out_reducer,
            ),
        ]


if __name__ == '__main__':
    MatMul.run()
