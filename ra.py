import ast, time, os
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sklearn.cluster import AffinityPropagation

from api import API

class RA:
    def __init__(self, kis_access_token=None):
        # KIS
        self.kis_appkey = os.getenv('kis_appkey')
        self.kis_appsecret = os.getenv('kis_appsecret')
        self.kis_cano = os.getenv('kis_cano')
        self.kis_api = API(os.getenv('kis_domain'))
        self.kis_access_token = kis_access_token \
            or self.get_kis_access_token()
        # Telegram
        self.tele_chat_id = os.getenv('tele_chat_id')
        self.tele_api = API(
            f'https://api.telegram.org/bot{os.getenv("tele_bot_token")}')
        # CONSTANT
        self.PERIODS = (2, 3, 5, 8, 13, 21)
        # --------
        self.send_msg(f'##############')

    def send_msg(self, text: str):
        self.tele_api.post('sendMessage', dict(
            chat_id=self.tele_chat_id, 
            text=text))

    # 계좌 관련

    def get_kis_access_token(self) -> str:
        payload = dict(
            grant_type='client_credentials',
            appkey=self.kis_appkey,
            appsecret=self.kis_appsecret)
        response = self.kis_api.post(
            path='oauth2/tokenP',
            payload=payload)
        return response.json().get('access_token')
        
    def get_kis_common_headers(self, tr_id) -> dict:
        return {
            'content-type': 'application/json; charset=utf-8',
            'authorization': f'Bearer {self.kis_access_token}',
            'appkey': self.kis_appkey,
            'appsecret': self.kis_appsecret,
            'tr_id': tr_id,
            'custtype': 'P'}        
    
    def get_kis_common_params(self, **kwargs) -> dict:
        return dict(
            CANO=self.kis_cano,
            ACNT_PRDT_CD='01',            
            **kwargs)

    def get_balance(self) -> pd.DataFrame:
        headers = self.get_kis_common_headers('TTTC8434R')
        params=self.get_kis_common_params(
            AFHR_FLPR_YN='N',
            OFL_YN='',
            INQR_DVSN='02',
            UNPR_DVSN='01',
            FUND_STTL_ICLD_YN='N',
            FNCG_AMT_AUTO_RDPT_YN='N',
            PRCS_DVSN='00',
            CTX_AREA_FK100='',
            CTX_AREA_NK100='')
        response = self.kis_api.get(
            path='uapi/domestic-stock/v1/trading/inquire-balance',
            params=params,
            headers=headers)
        data = response.json().get('output1')
        self.send_msg('[잔고 현황]')
        self.balance = self.format_balance_table(data)\
            [['상품번호', '상품명', '보유수량', '평가손익율']]
        for _, row in self.balance.iterrows():
            self.send_msg(
                f'종목코드 : {row["상품번호"]}\n'
                + f'종목명 : {row["상품명"]}\n'
                + f'보유수량 : {row["보유수량"]}주\n'
                + f'평가손익율 : {row["평가손익율"]:.2f}%')
        return self.balance
    
    def get_account_balance(self) -> pd.DataFrame:
        headers = self.get_kis_common_headers('CTRP6548R')
        params=self.get_kis_common_params(
            INQR_DVSN_1='',
            BSPR_BF_DT_APLY_YN='')
        response = self.kis_api.get(
            path='uapi/domestic-stock/v1/trading/inquire-account-balance',
            params=params,
            headers=headers)
        data = response.json().get('output1')
        self.account_balance = self.format_account_balance_table(data)
        self.send_msg('[계좌 현황]')
        for idx, row in self.account_balance.iterrows():
            self.send_msg(
                f'자산명 : {idx}\n'
                + f'평가금액 : {row["평가금액"]:,}원\n'
                + f'평가손익금액 : {int(row["평가손익금액"]):,}원\n'
                + f'전체비중율 : {row["전체비중율"]:.2f}%')
        return self.account_balance
    
    @classmethod
    def format_balance_table(cls, data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df.columns = [
            '상품번호', '상품명', '매매구분명', '전일매수수량', '전일매도수량',
            '금일매수수량', '금일매도수량', '보유수량', '주문가능수량', '매입평균가격',
            '매입금액', '현재가', '평가금액', '평가손익금액', '평가손익율',
            '평가수익율', '대출일자', '대출금액', '대주매각대금', '만기일자',
            '등락율', '전일대비증감', '종목증거금율명', '보증금율명', '대용가격',
            '주식대출단가']
        df = df.loc[:, [
            '상품번호', '상품명', '보유수량', '매입평균가격', '매입금액',
            '현재가', '평가금액', '평가손익금액', '평가손익율']]
        df = df.astype({
            '보유수량': int,
            '매입평균가격': float,
            '매입금액': int,
            '현재가': int,
            '평가금액': int,
            '평가손익금액': int,
            '평가손익율': float,
        })
        df.query('보유수량 > 0', inplace=True)
        return df
    
    @classmethod
    def format_account_balance_table(cls, data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data).astype(float)
        df.index = ["주식", "펀드/MMW", "채권", "ELS/DLS", "WRAP",
                    "신탁/퇴직연금/외화신탁", "RP/발행어음", "해외주식", "해외채권", "금현물",
                    "CD/CP", "단기사채", "타사상품", "외화단기사채", "외화 ELS/DLS",
                    "외화", "예수금+CMA", "청약자예수금", "<합계>"]
        df.columns = ["매입금액", "평가금액", "평가손익금액",
                    "신용대출금액", "실제순자산금액", "전체비중율"]
        df = df.loc[:, ['평가금액', '평가손익금액', '전체비중율']]\
            .astype({'평가금액': int, '평가손익금액': int}).query('평가금액 > 0')
        return df

    # 가격 관련

    def get_etf_list(self):
        response = API('https://finance.naver.com')\
                    .get('api/sise/etfItemList.nhn')
        data = response.json().get('result').get('etfItemList')
        df = self.screen_etf_list_table(data)
        self.etf_list = df.copy()
        self.send_msg(f'[조건 만족 ETF 개수]\n{len(df)}개')
        return df
    
    @classmethod
    def screen_etf_list_table(cls, data: dict):
        df = pd.DataFrame(data).loc[:,
         ['itemcode', 'etfTabCode', 'itemname', 'quant', 'marketSum']]
        exclude_keywords = ['액티브', '레버리지', '2X', '인도']
        filtered_df = df[~df['itemname'].str.contains('|'.join(exclude_keywords)) &
                        (df['marketSum'] >= df.marketSum.median() * 2) &
                        (df['etfTabCode'] != 1)]
        filtered_df = filtered_df[
                        (filtered_df['quant'] >= filtered_df.quant.median() * 2)]
        filtered_df.reset_index(drop=True, inplace=True)
        return filtered_df

    @classmethod
    def get_price(cls, symbol):
        params = {
            'symbol': symbol,
            'requestType': '1',
            'startTime': '20230101',
            'endTime': '20991231',
            'timeframe': 'day',
        }
        response = API('https://api.finance.naver.com')\
                    .get('siseJson.naver', params=params)
        data = ast.literal_eval(response.text.replace('\n', ''))
        df = pd.DataFrame(data[1:], columns=data[0]).iloc[:, [2, 3, 4]]
        return symbol, df
    
    @contextmanager
    def measure_execution_time(self, msg):
        self.send_msg(f'[{msg}]')
        start_time = time.time()
        yield
        self.send_msg(f'작업소요시간 : {time.time() - start_time:.2f}초')

    def get_prices(self):
        with self.measure_execution_time('가격 데이터 불러오기 시작'):
            futures = []
            results = {}
            with ThreadPoolExecutor() as executor:
                for symbol in self.etf_list['itemcode']:
                    futures.append(executor.submit(self.get_price, symbol))
                
                for future in as_completed(futures):
                    symbol, df = future.result()
                    if not df.empty:
                        results[symbol] = df
        self.prices = results.copy()
        return results.copy()

    @classmethod
    def get_tr(cls, price):
        price['이전 종가'] = price['종가'].shift(1)
        tr = pd.DataFrame({
            'th': price[['고가', '이전 종가']].max(axis=1),
            'tl': price[['저가', '이전 종가']].min(axis=1)
        })
        tr['TR'] = tr['th'] - tr['tl']
        return tr['TR'].tail(21).reset_index(drop=True)

    def get_trs(self):
        trs = {}
        for symbol, df in self.prices.items():
            tr = self.get_tr(df)
            if len(tr) == 21:
                trs[symbol] = tr
        self.trs = trs.copy()
        return trs

    def get_corr_matrix(self):
        df_tr = pd.concat(self.trs.values(), axis=1)
        df_tr.columns = self.trs.keys()
        self.corr_matrix = df_tr.corr()
        return df_tr.corr()
    
    def perform_clustering(self):
        clustering = AffinityPropagation(affinity='precomputed')
        clustering_labels = clustering.fit_predict(self.corr_matrix) + 1
        self.clustering_labels = clustering_labels.copy()
        return clustering_labels

    def format_clustering_results(self):
        groups = pd.DataFrame(
            zip(self.corr_matrix.index, self.clustering_labels),
            columns=['Itemcode', 'Group'])
        names = pd.DataFrame(
            zip(self.etf_list['itemcode'], self.etf_list['itemname']),
            columns=['Itemcode', 'Itemname'])
        clusters = groups.merge(names, on='Itemcode')\
            .set_index(['Group', 'Itemcode']).sort_index()
        self.clusters = clusters.copy()
        for group, group_data in clusters.groupby('Group'):
            self.send_msg(
                f"[클러스터 {group}]\n{', '.join(group_data['Itemname'])}\n")
        return clusters
    
    def calculate_momentum(self, price):
        return sum([(price.iloc[-1] / price.iloc[-p] - 1)
                    for p in self.PERIODS]) / len(self.PERIODS)

    def calculate_risk(self, tr, price):
        return tr.ewm(max(self.PERIODS)).mean().iloc[-1] / price['종가'].iloc[-1]

    def get_momentum(self):
        self.clusters['Momentum'] = [
            self.calculate_momentum(self.prices[itemcode]['종가'])
                for itemcode in self.clusters.reset_index()['Itemcode']]
        max_momentum = self.clusters.iloc[
            self.clusters.reset_index()
            .groupby('Group')['Momentum'].idxmax()].copy()
        max_momentum['Risk'] = [
            self.calculate_risk(self.trs[itemcode], self.prices[itemcode])
                for itemcode in max_momentum.reset_index()['Itemcode']]
        self.momentum = max_momentum.sort_values('Momentum', ascending=False)
        self.send_msg('[클러스터 모멘텀 랭킹]')
        for idx, row in self.momentum.iterrows():
            self.send_msg(
                f'그룹명 : {idx[0]}\n'
                + f'종목코드 : {idx[1]}\n'
                + f'종목명 : {row["Itemname"]}\n'
                + f'모멘텀 : {row["Momentum"]:.4f}\n'
                + f'리스크 : {row["Risk"]:.4f}\n')
        return max_momentum.sort_values('Momentum', ascending=False)

    def get_position(self, risk=0.01, candidate=4):
        trading_balance = self.account_balance\
            .loc[['주식', 'RP/발행어음', '예수금+CMA'],'평가금액'].sum()
        self.momentum['Enter'] = (risk / self.momentum['Risk'])\
            .apply(lambda x: min(1, x))\
            .apply(lambda x: int(
                trading_balance / candidate * x // 10000 * 10000))
        self.position = self.momentum.query('Momentum > 0').head(candidate + 1)
        return self.momentum.query('Momentum > 0').head(candidate + 1)

    def get_dashboard(self):
        table = self.position.reset_index().merge(self.balance,
                how='left', left_on='Itemcode', right_on='상품번호')
        table['Own'] = table['상품번호'].apply(lambda x: 'X' if pd.isna(x) else 'O')
        table.drop(columns='상품번호', inplace=True)
        table = table.astype({'Enter':int})
        self.dashboard = table.set_index('Itemcode')\
            .loc[:, ['Itemname', 'Momentum', 'Risk', 'Enter', 'Own']]
        self.send_msg('[클러스터 모멘텀 TOP5 & 진입규모]')
        for idx, row in self.dashboard.iterrows():
            self.send_msg(
                f'종목코드 : {idx}\n'
                + f'종목명 : {row["Itemname"]}\n'
                + f'모멘텀 : {row["Momentum"]:.4f}\n'
                + f'리스크 : {row["Risk"]:.4f}\n'
                + f'포지션 : {row["Enter"]:,}원\n'
                + f'보유중 : {row["Own"]}')
        return self.dashboard