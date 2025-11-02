"""
Compare all viable feature combinations
"""
import pandas as pd

overlap = pd.read_csv('mscl_overlap_50cm.csv')

combos = [
    (['GRA', 'MS', 'NGR'], 'v1: GRA+MS+NGR (3 features)'),
    (['GRA', 'MS', 'NGR', 'RGB'], 'v2.6: GRA+MS+NGR+RGB (4 features, BEST ARI=0.258)'),
    (['GRA', 'MS', 'NGR', 'RSC'], 'RSC: GRA+MS+NGR+RSC (4 features)'),
    (['GRA', 'MS', 'NGR', 'RSC', 'MSP'], 'RSC+MSP: GRA+MS+NGR+RSC+MSP (5 features)'),
    (['GRA', 'MS', 'MSP', 'NGR'], 'MSP: GRA+MS+MSP+NGR (4 features, no color)'),
    (['GRA', 'MS', 'NGR', 'RGB', 'RSC'], 'RGB+RSC: All color (5 features)'),
]

print('='*90)
print('Feature Combination Comparison (50cm bins)')
print('='*90)
print()
print(f"{'Description':<60s} {'Bins':>10s} {'Boreholes':>10s} {'~20cm':>12s}")
print('-'*90)

for feats, desc in combos:
    m = overlap[feats[0]].copy()
    for f in feats[1:]:
        m = m & overlap[f]
    bins = m.sum()
    bh = overlap[m]['Borehole_ID'].nunique()
    bins_20cm = int(bins * 2.3)
    print(f'{desc:<60s} {bins:>10,} {bh:>10} {bins_20cm:>12,}')

print()
print('='*90)
print('Recommendation Tiers')
print('='*90)
print()
print('TIER 1 - Most Data (but no color):')
print('  GRA+MS+NGR (v1): 171,047 bins, 524 boreholes')
print('    → ~393,000 samples at 20cm')
print('    → No color info, lower ARI expected')
print()
print('TIER 2 - Balance of Data + Color:')
print('  GRA+MS+NGR+RSC: 168,962 bins, 519 boreholes')
print('    → ~388,600 samples at 20cm (+63% vs RGB)')
print('    → RSC color info (L*a*b*)')
print('    → RECOMMENDED TO TEST')
print()
print('  GRA+MS+NGR+RSC+MSP: 148,058 bins, 484 boreholes')
print('    → ~340,500 samples at 20cm (+43% vs RGB)')
print('    → RSC color + extra magnetic (MSP)')
print('    → 5 FEATURES - ALSO RECOMMENDED')
print()
print('TIER 3 - Current Best (limited by RGB):')
print('  GRA+MS+NGR+RGB (v2.6): 101,883 bins, 296 boreholes')
print('    → 238,506 samples at 20cm')
print('    → Best known ARI = 0.258')
print('    → Limited by RGB coverage')
print()
print('='*90)
print('KEY INSIGHT')
print('='*90)
print()
print('GRA+MS+NGR+RSC+MSP gives:')
print('  • +43% more training data than RGB (340K vs 238K samples)')
print('  • +64% more boreholes (484 vs 296)')
print('  • 5 features instead of 4')
print('  • Color information from RSC (L*a*b*)')
print('  • Dual magnetic susceptibility (MS loop + MSP point)')
print()
print('Worth testing if RSC color achieves similar/better ARI than RGB!')
