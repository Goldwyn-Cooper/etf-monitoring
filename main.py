from ra import RA

def handler(event=None, context=None):
    ra = RA()
    ra.get_balance()
    ra.get_account_balance()
    ra.get_etf_list()
    ra.get_prices()
    ra.get_trs()
    ra.get_corr_matrix()
    ra.perform_clustering()
    ra.format_clustering_results()
    ra.get_momentum()
    ra.get_position()
    ra.get_dashboard()

if __name__ == '__main__':
    handler()