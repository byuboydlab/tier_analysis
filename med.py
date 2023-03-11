# code specific to the medical supply chain paper with Kayvan

def get_df(cached=True,save=True, extra_tiers=False):
    if cached:
        df = pd.read_hdf('dat/vc_relationships.h5') 
    else:
        file_name='dat/med.xlsx'
        #file_name='dat/T10-9-8-7-6-5-4-3-2-1 transformed no formula.xlsx'
        #file_name='Autoindustry Tier 1-2-3-4-5 only suppliers.xlsx'

        df = pd.read_excel(file_name,sheet_name="Sheet1",engine='openpyxl')
        df = df.drop_duplicates(ignore_index=True)

        try:
            df=df[df['Relationship Type']=='Supplier']
            df.reset_index()
        except:
            pass

        # resolve NaNs for better typing
        for col in ['Source Country', 'Target Country', 'Source Name', 'Target Name', 'Source Industry', 'Target Industry', 'Source Private', 'Target Private']:
            try: # in case these columns are not there
                df[col] = df[col].astype(str)
            except:
                pass
        for col in ['Source Market Cap', 'Target Market Cap', 'Source Revenue', 'Target Revenue', 'Source Employees Global', 'Target Employees Global']:
            try: # in case these columns are not there
                df.loc[df[col] == '(Invalid Identifier)',col] = math.nan
                df[col]=df[col].astype(float)
            except:
                pass

        if save:
            df.to_hdf('dat/vc_relationships.h5',key='df')

    return df

def no_china_us_reachability(G,include_taiwan_hong_kong=False,prefix='.'):
    print("Removing all US-China supply chain links")

    # Get medical suppliers
    firms = G.vs
    firms['is_med'] = [False] * G.vcount()
    for x in G.es(tier = 1): x.target_vertex['is_med'] = True

    # define china
    china=['China']
    if include_taiwan_hong_kong:
        print("Including Taiwan and Hong Kong in China")
        china += ['Hong Kong','Taiwan']
    else:
        print("Excluding Taiwan and Hong Kong from China")

    # find chinese/us firms
    is_ch = lambda x : type(x['country'] == str) and (x['country'] in china)
    is_us = lambda x : x['country'] == 'United States'

    # thin graph
    G_thin = deepcopy(G)
    G_thin.delete_edges(
            G_thin.es.select(
                _between = [
                    G_thin.vs(is_ch),
                    G_thin.vs(is_us)]))
    #print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    med_suppliers = get_demand_nodes(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]
    u = [get_u(i.index,G_thin) for i in med_suppliers]

    reachable = pd.DataFrame([[cb(i.index,G,G_thin,tt,uu) for i,tt,uu in zip(med_suppliers,t,u)] for cb in callbacks],index=[cb.description for cb in callbacks]).transpose()
    reachable['country'] = G.vs([i.index for i in med_suppliers])['country']
    for cb in callbacks:
        reachable[cb.description] = reachable[cb.description].astype(cb.type)

    by_country = reachable.groupby('country').mean().reindex(['United States', 'China', 'Taiwan', 'Hong Kong'])

    os.makedirs(prefix + 'dat/',exist_ok=True)
    by_country.to_excel(prefix + '/dat/no_us_china'+ ('_incl_taiwan_hk' if include_taiwan_hong_kong else '') + '.xlsx')

    return by_country

def close_all_borders(G,prefix='.'):
    G_thin = deepcopy(G)
    G_thin.delete_edges(G_thin.es.select(lambda e : e.source_vertex['country'] != e.target_vertex['country']))

    med_suppliers = get_demand_nodes(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]
    u = [get_u(i.index,G_thin) for i in med_suppliers]

    reachable = pd.DataFrame([[cb(i.index,G,G_thin,tt,uu) for i,tt,uu in zip(med_suppliers,t,u)] for cb in callbacks],
            index=[cb.description for cb in callbacks]).transpose()
    reachable['country'] = G.vs([i.index for i in med_suppliers])['country']
    for cb in callbacks:
        reachable[cb.description] = reachable[cb.description].astype(cb.type)

    by_country = reachable.groupby('country').mean()

    by_country.to_excel(prefix+'/dat/close_all_borders.xlsx')

    return reachable

def industry_deletion_effects(G,prefix='.'):

    med_suppliers = get_demand_nodes(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]

    res = dict()
    for ind in set(G.vs['industry']):
        if type(ind) is not float:
            G_thin = deepcopy(G)
            G_thin.delete_vertices(G.vs(industry=ind))
            u = [get_u(i.index,G_thin) for i in med_suppliers]
            res[ind] = [np.mean([cb(None,G,G_thin,tt,uu) for tt,uu in zip(t,u)]) for cb in callbacks]

    res = pd.DataFrame(res,index=[cb.description for cb in callbacks]).transpose()
    res.to_excel(prefix + '/dat/industry_deletion_effects.xlsx')

    return res

def run_all_simulations(
        G=None,
        attacks=None,
        repeats=None,
        giant=True,
        rho_scales=None,
        software_compare=False,
        scales_simulations=True,
        tiers_simulations=True,
        borders=True,
        tiers=range(1,max_tiers+1),
        write_mode='w',
        prefix='',
        protected_countries = []):

    if G is None:
        G = directed_igraph(giant=giant)

    old_backend = matplotlib.backends.backend
    matplotlib.use('Agg') # non-interactive

    full_rho = np.linspace(.3,1,71)
    failure_scales = ['firm','country','industry','country-industry']
    if attacks is None:
        attacks = [random_thinning_factory, 
                partial(get_pagerank_attack,transpose=True, protected_countries = protected_countries), 
                partial(get_pagerank_attack, transpose=False, protected_countries = protected_countries), 
                partial(get_employee_attack, protected_countries = protected_countries)]
        attacks[1].description = 'Pagerank of transpose'
        attacks[2].description = 'Pagerank'
        attacks[3].description = 'Employees'

    max_repeats=100
    if repeats=='min':
        max_repeats=6

    if (repeats is None) or (repeats == 'min'):
        repeats = dict()
        for attack in attacks:
            for failure_scale in failure_scales:
                repeats[attack,failure_scale] = 1
                if failure_scale == 'industry':
                    repeats[attack,failure_scale] = min(max_repeats,24)
                if (attack == random_thinning_factory) or (attack == get_employee_attack):
                    repeats[attack,failure_scale] = max_repeats


    if rho_scales is None:
        rho_scales = [full_rho]#, np.linspace(.9,1,101)]#, np.linspace(.99,1,101),np.linspace(.999,1,101),np.linspace(.9999,1,101)]

    med_suppliers = [i.index for i in get_demand_nodes(G)]


    prefix=clean_prefix(prefix)
    os.makedirs(prefix + 'dat/',exist_ok=True)
    resfile = prefix + 'dat/all_results.h5'

    to_return = []
    if scales_simulations:
        res = pd.DataFrame()
        for rho in rho_scales:
            for failure_scale in failure_scales:
                for attack in attacks:
                    print(failure_scale + ' ' + (attack.description if attack else 'random') + ' scale ' + str(rho[0]) + ' ' + str(rho[-1]))
                    plt.clf()
                    avgs=failure_reachability(G,
                            rho=rho, 
                            targeted_factory=attack, 
                            plot=False,
                            repeats=repeats[(attack,failure_scale)], 
                            failure_scale=failure_scale,
                            med_suppliers=med_suppliers,
                            prefix=prefix)
                    res=res.append(avgs,ignore_index=True)
        res.to_hdf(resfile,key='scales',mode=write_mode)
        to_return.append(res)

        print('plotting')
        for failure_scale in failure_scales:
            res_temp = res[res['Failure scale'] == failure_scale]
            for rho in rho_scales:
                for metric in callbacks:
                    print(failure_scale + ' ' + str(rho[0]) + ' ' + metric.description)
                    plt.clf()
                    attack_compare_plot(res_temp,
                            failure_scale=failure_scale,
                            rho_scale=rho,
                            fname = prefix + 'im/attack_compare_' + str(rho[0]) + '_' + str(rho[-1]) + '/'\
                                    + metric.description.replace(' ','_').replace('.','').lower()\
                                    + '/' + failure_scale + '.svg',
                            save=True,
                            metric=metric.description)

    if tiers_simulations:
        res = pd.DataFrame()
        for failure_scale in failure_scales:
            for attack in attacks:
                print('compare tiers ' + attack.description.lower() + ' ' + failure_scale)
                plt.clf()
                res = res.append(compare_tiers(G,
                    rho=full_rho,
                    repeats=repeats[(attack,failure_scale)],
                    plot='save',
                    attack=attack,
                    failure_scale=failure_scale,
                    tier_range=tiers,
                    prefix=prefix),
                    ignore_index=True)
        res.to_hdf(resfile,key='tiers',
                mode=('a' if scales_simulations else write_mode))
        to_return.append(res)


    if software_compare:
        res = pd.DataFrame()
        graphs = (directed_igraph(no_software = True, giant=giant), 
                directed_igraph(no_software = False, giant=giant))
        for G, inclusive  in zip(graphs, (False, True)):
            for attack in attacks:
                print('Software ' + ('included' if inclusive else 'excluded') + ' ' + (attack.description if attack else 'random'))
                plt.clf()
                avgs=failure_reachability(G,
                        rho=full_rho, 
                        targeted_factory=attack, 
                        save_only=True,
                        repeats=repeats[(attack,failure_scale)], 
                        failure_scale='firm',
                        G_has_no_software_flag = (not inclusive),
                        prefix=prefix + 'software_compare')
                avgs['Software included'] = inclusive
                res=res.append(avgs,ignore_index=True)
        res.to_hdf(resfile,key='software',
                mode=('a' if scales_simulations or tiers_simulations else write_mode))
        res['Software included']=res['Software included'].astype(bool)
        to_return.append(res)

    if borders:
        print('no_china_us_reachability')
        no_china_us_reachability(G,include_taiwan_hong_kong=True,prefix=prefix)
        print('no_china_us_reachability')
        no_china_us_reachability(G,include_taiwan_hong_kong=False,prefix=prefix)
        print('close_all_borders')
        close_all_borders(G,prefix=prefix)
        print('industry_deletion_effects')
        industry_deletion_effects(G,prefix=prefix)
    
    matplotlib.use(old_backend) # non-interactive ('Qt5Agg' for me)

    return tuple(to_return)
