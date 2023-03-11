from med import *

import argparse

# Construct an argument parser
all_args = argparse.ArgumentParser()

# Add arguments to the parser
# 'firm','country','industry','country-industry'
all_args.add_argument("failure_scale")
all_args.add_argument("rho")  # full, zoomed
all_args.add_argument("attack")  # 'Random', etc.

args = vars(all_args.parse_args())

failure_scale = args['failure_scale']
full_rho = np.linspace(.3, 1, 71)
if args['rho'] == 'full':
    rho = full_rho
else:
    rho = np.linspace(.9, 1, 101)
attack = args['attack']

giant = True
write_mode = 'w'
prefix = 'temp'

G = directed_igraph(giant=giant)

old_backend = matplotlib.backends.backend
matplotlib.use('Agg')  # non-interactive

if attack == 'Random':
    attack = random_thinning_factory
elif attack == 'Pagerank of transpose':
    attack = get_pagerank_attack
elif attack == 'Pagerank':
    attack = get_pagerank_attack_no_transpose
else:
    attack = get_employee_attack

max_repeats = 100
repeats = 1
if failure_scale == 'industry':
    repeats = min(max_repeats, 24)
if (attack == random_thinning_factory) or (attack == get_employee_attack):
    repeats = max_repeats

med_suppliers = [i.index for i in get_med_suppliers(G)]

prefix = clean_prefix(prefix)
os.makedirs(prefix + 'dat/', exist_ok=True)
resfile = prefix + 'dat/all_results.h5'

to_return = []
res = pd.DataFrame()
plt.clf()
avgs = failure_reachability(G,
                            rho=rho,
                            targeted_factory=attack,
                            plot=False,
                            repeats=repeats,
                            failure_scale=failure_scale,
                            med_suppliers=med_suppliers,
                            prefix=prefix)
res = res.append(avgs, ignore_index=True)
res.to_hdf(resfile, key='scales/' + args[failure_scale] + '/'args['rho'] + '/' + args['attack'], mode=write_mode)
to_return.append(res)

matplotlib.use(old_backend)  # non-interactive ('Qt5Agg' for me)
