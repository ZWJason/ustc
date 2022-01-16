from tqdm import tqdm, trange
from time import sleep
from random import random, randint

# class TqdmExtraFormat(tqdm):
#     """Provides a `total_time` format parameter"""
#     @property
#     def format_dict(self):
#         d = super(TqdmExtraFormat, self).format_dict
#         total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
#         d.update(total_time=self.format_interval(total_time) + " in total")
#         return d

# for i in TqdmExtraFormat(
#       range(10), ascii=" █",
#       bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):
#       sleep(0.1)



with tqdm(total=10) as t:
    for i in range(10):
        # Description will be displayed on the left
        t.set_description('GEN %i' % i)
        # Postfix will be displayed on the right,
        # formatted automatically based on argument's datatype
        t.set_postfix(loss=random(), gen=randint(1,999), str='h',
                      lst=[1, 2])
        sleep(0.1)
        d = t.format_dict
        #print(d["elapsed"])