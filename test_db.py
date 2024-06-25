from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from ChatGLM3 import ChatGLM3


db = SQLDatabase.from_uri("sqlite:///博金杯比赛数据.db")
print(db.dialect)
print(db.get_usable_table_names())

info = db.get_table_info(['A股公司行业划分表', 'A股票日行情表', '基金份额持有人结构', '基金债券持仓明细', '基金可转债持仓明细', '基金基本信息', '基金日行情表', '基金股票持仓明细', '基金规模变动表', '港股票日行情表'])
print(info)
# content = db.run("SELECT * FROM A股公司行业划分表 LIMIT 3;")
# print(content)



llm = ChatGLM3()
llm.load_model('/mnt/vos-6j19uo2q/code/AliRAG/ChatGLM3/chatglm3-6b')
print(llm)


# 生成prompt
question = 'A股公司行业划分表前十条内容是什么？'


prompt = '''你现在是一名优秀的数据库工程师，现在请你将下面的查询直接转换为sql。
            我将给你数据库的表的结构，请你分析结构并且按照要求完成任务，下面是数据库的表的结构以及表中的一些数据。
            
     
            采用的数据库为sqlite，
            其中包括以下十张表：
            ['A股公司行业划分表', 'A股票日行情表', '基金份额持有人结构', '基金债券持仓明细', '基金可转债持仓明细', '基金基本信息', '基金日行情表', '基金股票持仓明细', '基金规模变动表', '港股票日行情表']

            其中每张表的结构如下所示，每张表后面为该表的前三条信息。
            CREATE TABLE "A股公司行业划分表" (
                    "股票代码" TEXT, 
                    "交易日期" TEXT, 
                    "行业划分标准" TEXT, 
                    "一级行业名称" TEXT, 
                    "二级行业名称" TEXT
            )

            /*
            3 rows from A股公司行业划分表 table:
            股票代码        交易日期        行业划分标准    一级行业名称    二级行业名称
            000065  20190115        中信行业分类    建筑    建筑施工Ⅱ
            300502  20190125        中信行业分类    通信    通信设备
            600481  20190120        中信行业分类    机械    专用机械
            */


            CREATE TABLE "A股票日行情表" (
                    "股票代码" TEXT, 
                    "交易日" TEXT, 
                    "昨收盘(元)" REAL, 
                    "今开盘(元)" REAL, 
                    "最高价(元)" REAL, 
                    "最低价(元)" REAL, 
                    "收盘价(元)" REAL, 
                    "成交量(股)" REAL, 
                    "成交金额(元)" REAL
            )

            /*
            3 rows from A股票日行情表 table:
            股票代码        交易日  昨收盘(元)      今开盘(元)      最高价(元)      最低价(元)      收盘价(元)      成交量(股)      成交金额(元)
            603665  20190131        28.32   27.5    28.09   25.49   27.28   1757953.0       48247793.0
            300180  20190104        11.07   10.99   11.25   10.98   11.16   2269600.0       25305087.0
            300071  20190115        3.87    3.87    3.96    3.83    3.93    17855227.0      69867668.0
            */


            CREATE TABLE "基金份额持有人结构" (
                    "基金代码" TEXT, 
                    "基金简称" TEXT, 
                    "公告日期" TIMESTAMP, 
                    "截止日期" TIMESTAMP, 
                    "机构投资者持有的基金份额" REAL, 
                    "机构投资者持有的基金份额占总份额比例" REAL, 
                    "个人投资者持有的基金份额" REAL, 
                    "个人投资者持有的基金份额占总份额比例" REAL, 
                    "定期报告所属年度" INTEGER, 
                    "报告类型" TEXT
            )

            /*
            3 rows from 基金份额持有人结构 table:
            基金代码        基金简称        公告日期        截止日期        机构投资者持有的基金份额        机构投资者持有的基金份额占总份额比例    个人投资者持有的基金份额        个人投资者持有的基金份额占总份额比例    定期报告所属年度        报告类型
            000006  西部利得量化成长混合A   2019-08-24 00:00:00     2019-06-30 00:00:00     10000600.0      7.24    128087037.15    92.76   2019    中期报告
            000028  华富安鑫债券    2019-08-29 00:00:00     2019-06-30 00:00:00     217513.79       0.25    88253498.14     99.75   2019    中期报告
            000030  长城核心优选灵活配置混合A       2019-08-27 00:00:00     2019-06-30 00:00:00     18574577.58     6.03    289425324.34    93.97   2019中期报告
            */


            CREATE TABLE "基金债券持仓明细" (
                    "基金代码" TEXT, 
                    "基金简称" TEXT, 
                    "持仓日期" TEXT, 
                    "债券类型" TEXT, 
                    "债券名称" TEXT, 
                    "持债数量" REAL, 
                    "持债市值" REAL, 
                    "持债市值占基金资产净值比" REAL, 
                    "第N大重仓股" INTEGER, 
                    "所在证券市场" TEXT, 
                    "所属国家(地区)" TEXT, 
                    "报告类型" TEXT
            )

            /*
            3 rows from 基金债券持仓明细 table:
            基金代码        基金简称        持仓日期        债券类型        债券名称        持债数量        持债市值        持债市值占基金资产净值比第N大重仓股     所在证券市场    所属国家(地区)  报告类型
            010005  鹏扬现金通利货币E       20210331        超短期融资券    21华能水电SCP002        200000.0        20000180.24     0.0253  9       银行间市场      中华人民共和国  季报
            007659  博时富汇纯债3个月定期开放债券   20191231        公司债券        16恒健01        800000.0        79832000.0      0.0825  1       上海证券交易所  中华人民共和国  年报(含半年报)
            007659  博时富汇纯债3个月定期开放债券   20200331        公司债券        16恒健01        800000.0        80536000.0      0.0832  2       上海证券交易所  中华人民共和国  季报
            */


            CREATE TABLE "基金可转债持仓明细" (
                    "基金代码" TEXT, 
                    "基金简称" TEXT, 
                    "持仓日期" TEXT, 
                    "对应股票代码" TEXT, 
                    "债券名称" TEXT, 
                    "数量" REAL, 
                    "市值" REAL, 
                    "市值占基金资产净值比" REAL, 
                    "第N大重仓股" INTEGER, 
                    "所在证券市场" TEXT, 
                    "所属国家(地区)" TEXT, 
                    "报告类型" TEXT
            )

            /*
            3 rows from 基金可转债持仓明细 table:
            基金代码        基金简称        持仓日期        对应股票代码    债券名称        数量    市值    市值占基金资产净值比    第N大重仓股     所在证券市场    所属国家(地区)  报告类型
            006650  招商安庆债券    20191231        300568  星源转债        1815.0  225989.4        0.0013  36      深圳证券交易所  中华人民共和国  季报
            006006  诺安鼎利混合C   20190930        300568  星源转债        6781.0  776260.8        0.0062  29      深圳证券交易所  中华人民共和国  季报
            006467  浦银安盛双债增强债券C   20191231        300568  星源转债        4987.0  620850.0        0.003   19      深圳证券交易所  中华人民共和国季报
            */


            CREATE TABLE "基金基本信息" (
                    "基金代码" TEXT, 
                    "基金全称" TEXT, 
                    "基金简称" TEXT, 
                    "管理人" TEXT, 
                    "托管人" TEXT, 
                    "基金类型" TEXT, 
                    "成立日期" TEXT, 
                    "到期日期" TEXT, 
                    "管理费率" TEXT, 
                    "托管费率" TEXT
            )

            /*
            3 rows from 基金基本信息 table:
            基金代码        基金全称        基金简称        管理人  托管人  基金类型        成立日期        到期日期        管理费率        托管费率
            000006  西部利得量化成长混合型发起式证券投资基金A类     西部利得量化成长混合A   西部利得基金管理有限公司        中国农业银行股份有限公司混合型  20190319        30001231        1.2%    0.1%
            000028  华富安鑫债券型证券投资基金      华富安鑫债券    华富基金管理有限公司    上海浦东发展银行股份有限公司    债券型  20190612        300012310.7%    0.2%
            000030  长城核心优选灵活配置混合型证券投资基金A类       长城核心优选灵活配置混合A       长城基金管理有限公司    中国建设银行股份有限公司混合型  20190524        30001231        1.2%    0.2%
            */


            CREATE TABLE "基金日行情表" (
                    "基金代码" TEXT, 
                    "交易日期" TEXT, 
                    "单位净值" REAL, 
                    "复权单位净值" REAL, 
                    "累计单位净值" REAL, 
                    "资产净值" REAL
            )

            /*
            3 rows from 基金日行情表 table:
            基金代码        交易日期        单位净值        复权单位净值    累计单位净值    资产净值
            007120  20210120        2.162   2.162   2.162   3282338328.05
            006845  20210603        1.2599  1.2599  1.2599  2692024.53
            010157  20211221        1.1338  1.1338  1.1338  137142492.09
            */


            CREATE TABLE "基金股票持仓明细" (
                    "基金代码" TEXT, 
                    "基金简称" TEXT, 
                    "持仓日期" TEXT, 
                    "股票代码" TEXT, 
                    "股票名称" TEXT, 
                    "数量" REAL, 
                    "市值" REAL, 
                    "市值占基金资产净值比" REAL, 
                    "第N大重仓股" INTEGER, 
                    "所在证券市场" TEXT, 
                    "所属国家(地区)" TEXT, 
                    "报告类型" TEXT
            )

            /*
            3 rows from 基金股票持仓明细 table:
            基金代码        基金简称        持仓日期        股票代码        股票名称        数量    市值    市值占基金资产净值比    第N大重仓股     所在证券市场    所属国家(地区)  报告类型
            007484  信澳核心科技混合A       20201231        600563  法拉电子        151369.0        16279735.95     0.0257  4       上海证券交易所  中华人民共和国  季报
            006713  前海开源MSCI中国A股消费指数C    20210630        002216  三全食品        21770.0 358551.9        0.0021  2       深圳证券交易所  中华人民共和国  年报(含半年报)
            008935  大成科技消费股票C       20210630        300991  创益通  335.0   9662.5  0.0     93      深圳证券交易所  中华人民共和国  年报(含半年报)
            */


            CREATE TABLE "基金规模变动表" (
                    "基金代码" TEXT, 
                    "基金简称" TEXT, 
                    "公告日期" TIMESTAMP, 
                    "截止日期" TIMESTAMP, 
                    "报告期期初基金总份额" REAL, 
                    "报告期基金总申购份额" REAL, 
                    "报告期基金总赎回份额" REAL, 
                    "报告期期末基金总份额" REAL, 
                    "定期报告所属年度" INTEGER, 
                    "报告类型" TEXT
            )

            /*
            3 rows from 基金规模变动表 table:
            基金代码        基金简称        公告日期        截止日期        报告期期初基金总份额    报告期基金总申购份额    报告期基金总赎回份额    报告期期末基金总份额    定期报告所属年度        报告类型
            000028  华富安鑫债券    2019-04-20 00:00:00     2019-03-31 00:00:00     344550555.65    1811778.99      18997687.33     327364647.31    2019基金定期报告
            000030  长城核心优选灵活配置混合A       2019-04-22 00:00:00     2019-03-31 00:00:00     1686849451.21   2592968.41      97884441.21     1591557978.41   2019    基金定期报告
            000037  广发景宁纯债债券A       2019-04-22 00:00:00     2019-03-31 00:00:00     524340286.96    3272117.58      167762155.66    359850248.88   2019     基金定期报告
            */


            CREATE TABLE "港股票日行情表" (
                    "股票代码" TEXT, 
                    "交易日" TEXT, 
                    "昨收盘(元)" REAL, 
                    "今开盘(元)" REAL, 
                    "最高价(元)" REAL, 
                    "最低价(元)" REAL, 
                    "收盘价(元)" REAL, 
                    "成交量(股)" REAL, 
                    "成交金额(元)" REAL
            )

            /*
            3 rows from 港股票日行情表 table:
            股票代码        交易日  昨收盘(元)      今开盘(元)      最高价(元)      最低价(元)      收盘价(元)      成交量(股)      成交金额(元)
            47 HK   20190125        0.162   0.16    0.164   0.16    0.163   68000.0 10908.0
            08210   20190107        0.65    0.64    0.66    0.58    0.63    10460000.0      6488500.0
            2280 HK 20190109        4.19    4.19    4.31    4.24    4.29    1387500.0       5950695.0
            */
            
            有以下几点要求：
            首先直接给出sql,无需中间步骤及多余回答
            如果涉及到查询，如果没有要求排序，则无需排序，直接查询即可。
            上面是几点要求，请你将下面的查询转换为sql：
            ''' + question
print(f'生成的prompt为：{prompt}')

# llm响应，给出回答
response = llm.qa(prompt)


query = response[0]
print(f'llm生成的回答为：{response}\n')
# print("")
print(f'从回答中抽取的sql为：{query}\n')


result = db.run(query)
print(f'从数据库中查询出的数据为{result}')

