#%%
# TODO: CHANGE THIS FILE NAME TO DMA_project2_team##.py
# EX. TEAM 1 --> DMA_project2_team01.py


# TODO: IMPORT LIBRARIES NEEDED FOR PROJECT 2
import mysql.connector
import os
import csv
from sklearn.preprocessing import OneHotEncoder
import surprise
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise.model_selection.search import GridSearchCV
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import tree
import graphviz
from mlxtend.frequent_patterns import association_rules, apriori
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz



np.random.seed(0)

# TODO: CHANGE GRAPHVIZ DIRECTORY
# If you installed graphviz with the command conda install python-graphviz at the anaconda prompt, you would not need the following procedure.
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz2.47.1/bin/'
# os.environ["PATH"] += os.pathsep + '/usr/local/Cellar/graphviz/2.47.1/bin/'  # for MacOS

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz2.47.1/bin/'



# TODO: CHANGE MYSQL INFORMATION, team number
HOST = 'localhost'
USER = 'root'
PASSWORD = 'password'
SCHEMA = 'DMA_TEAM09'
team = 9

#%%
# PART 1: Decision tree 

"""
part1() 코드가 작동되지 않을 시 주의사항
windows 기준, 코드 실행 전에, C:\ProgramData\MySQL\MySQL Server 8.0\my.ini 파일을 열어서,
[mysqld] 아래 칸에 local_infile=1 을 입력 후 저장합니다.

또한 mysql workbench 및 터미널 콘솔에서 mysql 로그인 후 다음 쿼리를 실행합니다
mysql> set global local_infile=1;

그 후 services.msc에서 MySQL80을 재시작합니다.

또한 이 파이썬 코드와 같은 위치에 best_restaurant.txt가 포함된 dataset 폴더가 존재해야 합니다.

그래도 작동하지 않을 시 바로 아래 cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD, port = 3306, allow_local_infile=True)
에서 allow_local_infile=True 을 추가합니다.
"""


def part1():
    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD, port = 3306, allow_local_infile=True)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)

    # TODO: Requirement 1-1. MAKE best_item column
    cursor.execute("SHOW COLUMNS FROM Restaurant LIKE 'best_restaurant';")
    result = cursor.fetchone()
    if result:
        cursor.execute("ALTER TABLE Restaurant DROP COLUMN best_restaurant;")
    cursor.execute("SHOW COLUMNS FROM Restaurant LIKE 'best_restaurant_id';")
    result = cursor.fetchone()
    if result:
        cursor.execute("ALTER TABLE Restaurant DROP COLUMN best_restaurant_id;")

    cursor.execute("ALTER TABLE Restaurant ADD best_restaurant TINYINT(1) DEFAULT 0;")

    cursor.execute("ALTER TABLE Restaurant ADD best_restaurant_id VARCHAR(255);")
    cnx.commit()
   # TODO: Requirement 1-1.
    cursor.execute("DROP TABLE IF EXISTS temp_best_restaurant;")
    cursor.execute("CREATE TABLE temp_best_restaurant (temp_id VARCHAR(255));")
    cursor.execute(
        "LOAD DATA LOCAL INFILE './dataset/best_restaurant.txt' INTO TABLE temp_best_restaurant;")
    cursor.execute("UPDATE temp_best_restaurant SET temp_id = left(temp_id,22);")
    cursor.execute('''UPDATE Restaurant R, temp_best_restaurant T
                      SET R.best_restaurant_id = temp_id
                      WHERE lower(left(R.restaurant_id,22)) = lower(T.temp_id);''')
    cursor.execute(
        '''UPDATE Restaurant SET best_restaurant = 1 WHERE best_restaurant_id = restaurant_id;''')
    cnx.commit()
    cursor.execute("ALTER TABLE review MODIFY COLUMN taste_score DECIMAL(11,1);")
    cursor.execute("ALTER TABLE review MODIFY COLUMN service_score DECIMAL(11,1);")
    cursor.execute("ALTER TABLE review MODIFY COLUMN mood_score DECIMAL(11,1);")

    cursor.execute("""
        UPDATE Review 
        SET taste_score = total_score, 
            service_score = total_score, 
            mood_score = total_score 
        WHERE taste_score = 0 AND service_score = 0 AND mood_score = 0 AND total_score != 0;
    """)
    cnx.commit()

    # TODO: Requirement 1-2. WRITE MYSQL QUERY AND EXECUTE. SAVE to .csv file
    # TODO: Requirement 1-2.
    cursor.execute('''SELECT 
    r.restaurant_id,
    r.best_restaurant,
    AVG(re.total_score) AS avg_total_score,
    AVG(re.taste_score) AS avg_taste_score,
    AVG(re.service_score) AS avg_service_score,
    AVG(re.mood_score) AS avg_mood_score,
    COUNT(re.restaurant) AS num_of_reviews,
    (SELECT COUNT(c.user_id) FROM Collection c WHERE c.restaurant_id = r.restaurant_id) AS num_of_collections,
    ca.name AS category_name
FROM 
    Restaurant AS r
LEFT JOIN 
    Review AS re ON r.restaurant_id = re.restaurant
LEFT JOIN 
    Category AS ca ON r.category = ca.category_id
GROUP BY 
    r.restaurant_id;
            ''')
    data = cursor.fetchall()
    print(data)
    with open("./DMA_project2_team09_part1.csv", 'w', encoding='utf-8-sig', newline='') as f_handle:
        writer = csv.writer(f_handle)
        writer.writerow(
        ['restaurant_id', 'best_restaurant', 'avg_total_score', 'avg_taste_score', 'avg_service_score', 'avg_mood_score',
             'num_of_reviews', 'num_of_collections', 'category_name'])
        for row in data:
          writer.writerow(row)

    # TODO: Requirement 1-3. MAKE AND SAVE DECISION TREE
    # gini file name: DMA_project2_team##_part1_gini.pdf
    # entropy file name: DMA_project2_team##_part1_entropy.pdf

    df = pd.read_csv("./DMA_project2_team09_part1.csv")
    df[['avg_total_score', 'avg_taste_score', 'avg_service_score', 'avg_mood_score']] = df[
        ['avg_total_score', 'avg_taste_score', 'avg_service_score', 'avg_mood_score']].fillna(3)

    encoder = OneHotEncoder(sparse_output=False)
    encoded_categories = encoder.fit_transform(df[['category_name']])
    category_columns = encoder.get_feature_names_out(['category_name'])
    df_encoded = pd.DataFrame(encoded_categories, columns=category_columns)
    print(df_encoded)
    df = pd.concat([df.drop('category_name', axis=1), df_encoded], axis=1)

    features = df.drop(['best_restaurant', 'restaurant_id'], axis=1)  # ?삁痢≪쓣 ?쐞?븳 ?듅?꽦 ?뜲?씠?꽣, restaurant_id ?젣嫄?
    classes = df['best_restaurant']
    DT_gini = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=10, max_depth=5)
    DT_gini.fit(X=features, y=classes)
    print(DT_gini.get_params())
    print('-----------------------------------------------')
    # make entropy tree
    DT_entropy = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=5)
    DT_entropy.fit(X=features, y=classes)
    print(DT_entropy.get_params())
    print('-----------------------------------------------')

    def pdf_export_font_change(decision_tree, file_name_prefix, feature_names):
        dot_file_path = f'./{file_name_prefix}.dot'
        export_graphviz(decision_tree, out_file=dot_file_path,
                        feature_names=feature_names, class_names=['normal', 'BEST'], filled=True)
        font_change_dot_file(dot_file_path)
        return dot_file_path

    def font_change_dot_file(dot_path):
        modified_lines = []
        with open(dot_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            if 'fontname=' in line:
                line = line.replace('fontname="helvetica"', 'fontname="Sans"')
            modified_lines.append(line)

        with open(dot_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)

    def export_pdf_from_dot(dot_path, output_pdf_path):
        dot_graph = graphviz.Source.from_file(dot_path)
        dot_graph.render(filename=output_pdf_path, format='pdf', cleanup=True)

    dot_path_gini = pdf_export_font_change(DT_gini, 'DMA_project2_team09_part1_gini', features.columns)
    output_path_gini = "./DMA_project2_team09_part1_gini"
    export_pdf_from_dot(dot_path_gini, output_path_gini)

    dot_path_entropy = pdf_export_font_change(DT_entropy, 'DMA_project2_team09_part1_entropy', features.columns)
    output_path_entropy = "./DMA_project2_team09_part1_entropy"
    export_pdf_from_dot(dot_path_entropy, output_path_entropy)

    # TODO: Requirement 1-4. Don't need to append code for 1-4
    #using df from 1-3
    features = df.drop(['best_restaurant', 'restaurant_id', 'avg_service_score', 'avg_mood_score'], axis=1)
    classes = df['best_restaurant']
    DT_gini = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=20, max_depth=3)
    DT_gini.fit(X=features, y=classes)
    print(DT_gini.get_params())
    print('-----------------------------------------------')

    dot_path_gini = pdf_export_font_change(DT_gini, 'DMA_project2_team09_part1_gini_modified', features.columns)
    output_path_gini = "./DMA_project2_team09_part1_gini_modified"
    export_pdf_from_dot(dot_path_gini, output_path_gini)
    cursor.close()



#%%
# PART 2: Association analysis
def part2():
    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD, port = 3306)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)

    # TODO: Requirement 2-1. CREATE VIEW AND SAVE to .csv file
    cursor.execute('''CREATE VIEW restaurant_score AS
                   SELECT 
                    RES.restaurant_id AS restaurant_id,
                    RES.restaurant_name AS restaurant_name,
                    COALESCE(COL.num_collection,0) AS num_collection,
                    COALESCE(REV.num_review,0) AS num_review,
                    COALESCE(CAT.num_category_restaurant,0) AS num_category_restaurant,
                    COALESCE(SUM(REVRE.count_revisit - 1),0) AS num_revisit,
                    COALESCE(SUM(FOL.num_follower*REVRE.count_revisit), 0) AS influential_review_score,
                    (COALESCE(COL.num_collection,0)*5+COALESCE(REV.num_review,0)+COALESCE(CAT.num_category_restaurant,0)+COALESCE(SUM(FOL.num_follower*REVRE.count_revisit), 0)+COALESCE(SUM(REVRE.count_revisit - 1),0)*5) AS score
                    FROM 
                        Restaurant AS RES
                    LEFT JOIN 
                        (SELECT restaurant_id, COUNT(*) AS num_collection FROM Collection GROUP BY restaurant_id) AS COL ON RES.restaurant_id = COL.restaurant_id
                    LEFT JOIN 
                        (SELECT restaurant, COUNT(*) AS num_review FROM Review GROUP BY restaurant) AS REV ON RES.restaurant_id = REV.restaurant
                    JOIN 
                        (SELECT category, COUNT(*) AS num_category_restaurant FROM Restaurant GROUP BY category) AS CAT ON RES.category = CAT.category
                    LEFT JOIN 
                        (SELECT restaurant, user_id, COUNT(*) AS count_revisit FROM Review GROUP BY restaurant, user_id) AS REVRE ON RES.restaurant_id = REVRE.restaurant
                    LEFT JOIN 
                        (SELECT followee_id, COUNT(*) AS num_follower FROM Follow GROUP BY followee_id) AS FOL ON REVRE.user_id = FOL.followee_id AND num_follower>=3
                    GROUP BY 
                        RES.restaurant_id
                    ORDER BY
                        score DESC
                   
                   LIMIT 300;
                   ''')
    cursor.execute('select * from restaurant_score;')  
    pd_restaurant_score = pd.DataFrame(cursor.fetchall())  
    pd_restaurant_score.columns = cursor.column_names
    pd_restaurant_score.to_csv('DMA_project2_team%02d_part2_restaurant.csv' % team, sep=',', na_rep='NaN', index=False, encoding="utf-8-sig")    

    # TODO: Requirement 2-1.

    
    # TODO: Requirement 2-2. CREATE 2 VIEWS AND SAVE partial one to .csv file
    for row in cursor:
        print(row)    
    cursor.execute('''CREATE VIEW user_restaurant_IntDegree AS
                    SELECT 
                        U.user_id AS user,
                        R.restaurant_id AS restaurant,    
                        FLOOR(R4.res_avg/R5.user_avg+G1.rev_num+T1.col_num) AS IntDegree
                        FROM
                            Restaurant AS R
                        CROSS join
                            User as U
                        JOIN
                            (SELECT restaurant,user_id, AVG(total_score) AS res_avg FROM Review GROUP BY restaurant,user_id) AS R4 ON R4.restaurant=R.restaurant_id AND R4.user_id = U.user_id
                        JOIN
                            (SELECT user_id, AVG(total_score) AS user_avg FROM Review GROUP BY user_id) AS R5 ON R5.user_id=U.user_id
                        JOIN
                            (select G.user_id, G.category, COUNT(distinct G.restaurant_id) AS rev_num FROM (select RE.user_id, R3.restaurant_id, R3.category FROM Review AS RE JOIN Restaurant AS R3 on R3.restaurant_id=RE.restaurant) AS G group by G.user_id, G.category HAVING rev_num>0) AS G1 ON G1.user_id=U.user_id AND G1.category=R.category
                        JOIN
                            (select T.user_id, T.category, COUNT(distinct T.restaurant_id) AS col_num  FROM (select COL.user_id, R2.restaurant_id, R2.category FROM Collection AS COL JOIN Restaurant AS R2 on R2.restaurant_id=COL.restaurant_id ) AS T group by T.user_id, T.category) AS T1 ON T1.category=R.category AND T1.user_id=U.user_id

                        JOIN
                            (select restaurant_id FROM restaurant_score) AS RS ON RS.restaurant_id=R.restaurant_id;

                                                
                     ''')
    cursor.execute('''CREATE VIEW partial_user_restaurant_IntDegree AS
                    SELECT 
                        U.user AS user,
                        U.restaurant AS restaurant,
                        U.IntDegree AS IntDegree
                    FROM 
                        user_restaurant_IntDegree AS U
                    JOIN
                        (SELECT user, COUNT(*) AS num FROM user_restaurant_IntDegree group by user) AS UU on UU.user=U.user AND num>=25;  

                   ''')
    cnx.close()

    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD, port = 3306)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)

    cursor.execute('select * from partial_user_restaurant_IntDegree;')  
    
    pd_partial_user_restaurant_IntDegree = pd.DataFrame(cursor.fetchall())  
    pd_partial_user_restaurant_IntDegree.columns = cursor.column_names
    pd_partial_user_restaurant_IntDegree.to_csv('DMA_project2_team%02d_part2_UAI.csv' % team, sep=',', na_rep='NaN', index=False, encoding="utf-8-sig")
    # TODO: Requirement 2-2.

    
    # TODO: Requirement 2-3. MAKE HORIZONTAL VIEW
    # file name: DMA_project2_team##_part2_horizontal.pkl
    restaurant_id = []  
    cursor.execute('select restaurant from partial_user_restaurant_IntDegree;')
    while True:
        row = cursor.fetchone()  
        if row == None:
            break  
        if row[0] not in restaurant_id:
            restaurant_id.append(row[0])  
    sql1 = ''
    
    for i in restaurant_id:
        sql = 'max(if(restaurant=\'{id_1}\', 1, 0)) as \'{id_2}\''.format(id_1=i, id_2=i)
        sql1 = sql1 + ', ' + sql
    print('aaaaaaaaaa')

    cursor.execute('''select user
        {} from (select user, restaurant from partial_user_restaurant_IntDegree) as a
        group by user;
        '''.format(sql1))  

    hor_view = pd.DataFrame(cursor.fetchall())  
    hor_view.columns = cursor.column_names  
    hor_view = hor_view.set_index('user')  
    hor_view.to_pickle('DMA_project2_team%02d_part2_horizontal.pkl' % team)  
    
    # TODO: Requirement 2-3.

    
    # TODO: Requirement 2-4. ASSOCIATION ANALYSIS
    # filename: DMA_project2_team##_part2_association.pkl (pandas dataframe)
    hor_view = hor_view.replace(0, False)
    hor_view = hor_view.replace(1, True)

    frequent_itemsets = apriori(hor_view, min_support=0.2, use_colnames=True) 
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=2)  
    rules.to_pickle('DMA_project2_team%02d_part2_association.pkl' % team)  




    df = pd.read_pickle(f'DMA_project2_team09_part2_association.pkl')
    # Total number of association rules
    print(f"Total number of association rules: {len(df)}")
    # Top 10 rules sorted by lift
    print("\nTop 10 rules sorted by lift:")
    print(df.sort_values('lift', ascending=False).head(10))
    # Quantitative Analysis
    print("\nQuantitative Analysis:")
    # Descriptive statistics of support, confidence, and lift
    print(df[['support', 'confidence', 'lift']].describe())

    # TODO: Requirement 2-4.
    cursor.close()

#%%
# TODO: assign restaurant_file_path with path of 'restaurant.csv'
def get_name_of_restaurant(rid):
    restaurant_file_path = './dataset/Restaurant.csv'
    restaurant_list = pd.read_csv(restaurant_file_path,dtype=object)
    return restaurant_list[restaurant_list['restaurant_id']==rid]['restaurant_name']



#%%

# TODO: Requirement 3-1. WRITE get_top_n
def get_top_n(algo, testset, id_list, n, user_based=True):
    results = defaultdict(list)
    testset = list(filter(lambda x: len(x[1]) < 100, testset))
    testset_id = []
    if user_based:
        # TODO: From the testset data, save only the data where the user id is in id_list to testset_id
        # Hint: testset is a list with tuples of (user_id, restaurant_id, default_rating)
        testset_id = [x for x in testset if x[0] in id_list]
        # TODO
        
        predictions = algo.test(testset_id)
        for uid, rid, true_r, est, _ in predictions:
            # TODO: results is a dictionary with user_id as key and a list of tuples [(restaurant_id, estimated_rating)] as value
            results[uid].append((rid, est))
            # TODO
    else:
        # TODO: From the testset data, save only the data where the restaurant id is in id_list to testset_id
        # Hint: testset is a list with tuples of (user_id, restaurant_id, default_rating)
        testset_id = [x for x in testset if x[1] in id_list]
        # TODO
        
        predictions = algo.test(testset_id)
        for uid, rid, true_r, est, _ in predictions:
            # TODO: results is a dictionary with restaurant_id as key and a list of tuples [(user_id, estimated_rating)] as value
            results[rid].append((uid, est))
            # TODO

    for id_, ratings in results.items():
        # TODO: Sort by rating and keep only the top-n
        ratings.sort(key=lambda x: x[1], reverse=True)
        results[id_] = ratings[:n]
        # TODO

    return results


#%%
# PART 3. Requirement 3-2, 3-3, 3-4
def part3():
    # TODO: assign file_path with path of 'Rating.csv'
    file_path = './dataset/Rating.csv'
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0.5, 5.0), skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)

    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()

    uid_list = ['sHp92HPhfEused1lDyY5FA', 'kN5wr4aaOJWIRvbJZamO6Q',
       'WHA89VBJuWWkRAID0C6zCw', '2bZ4xxQGSln_LpRVG0smoQ',
       'iqtGbG_2mBe_TVWDQZJmNQ']
    # TODO - set algorithm for 3-2-1
    # KNNBasic, similarity : cosine
    sim_options = {'name': 'cosine', 'user_based': True}
    algo = surprise.KNNBasic(sim_options=sim_options)
    # TODO
    
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('3-2-1.txt', 'w', encoding='utf-8') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for rid, score in ratings:
                f.write('Restaurant name %s\n\tscore %s\n' % (get_name_of_restaurant(rid), str(score)))
            f.write('\n')
    print('3-2-1 ended')


    # TODO - set algorithm for 3-2-2
    # KNNWithMeans, similarity : pearson
    sim_options = {'name': 'pearson', 'user_based': True}
    algo = surprise.KNNWithMeans(sim_options=sim_options)

    # TODO
    
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('3-2-2.txt', 'w', encoding='utf-8') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for rid, score in ratings:
                f.write('Restaurant name %s\n\tscore %s\n' % (get_name_of_restaurant(rid), str(score)))
            f.write('\n')
    print('3-2-2 ended')



    # TODO - 3-2-3. Best Model
    # 1. lists of algorithms

    # msd, cosine, Pearson, Pearson_baseline similarity options are used.
    # function => KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline

    # reason for KFold : when the number of data is small, the result of the model may vary depending on the data division.
    # Therefore, we use KFold to divide the data into 5 parts and evaluate the model 5 times.
    # The average of the evaluation results is used as the final evaluation result.

    # algorithms setting,     # sim_options and parameter setting
    algorithms = [surprise.KNNBasic, surprise.KNNWithMeans, surprise.KNNWithZScore, surprise.KNNBaseline]
    sim_options = [{'name': 'msd','user_based': True}, {'name': 'cosine','user_based': True}, {'name': 'pearson','user_based': True}, {'name': 'pearson_baseline','user_based': True}]
    np.random.seed(0)
    kf = KFold(n_splits=5)
    best_algo = None
    best_sim_option = None
    best_score = float('inf')

    # Cross valiation to find the best algorithm and similarity option
    for algo in algorithms:
        for sim_option in sim_options:
            algo_instance = algo(sim_options=sim_option)
            rmse_scores = []
            for trainset, testset in kf.split(data):
                algo_instance.fit(trainset)
                predictions = algo_instance.test(testset)
                rmse = surprise.accuracy.rmse(predictions, verbose=True)
                rmse_scores.append(rmse)
            mean_rmse = np.mean(rmse_scores)
            if mean_rmse < best_score:
                best_algo = algo
                best_sim_option = sim_option
                best_score = mean_rmse

    # Prints the best algorithm, similarity option, and the average RMSE
    print(f"Best Algorithm: {best_algo.__name__}")
    print(f"Best Similarity Option: {best_sim_option}")
    print(f"Best Score (Average RMSE): {best_score}")


    # best algorithm and, result : 
    ''' 
    Best Algorithm: KNNBasic
    Best Similarity Option: {'name': 'pearson_baseline', 'user_based': True}
    Best Score (Average RMSE): 0.9305325338631297
    '''
    
    best_algo_ib = best_algo(sim_options=best_sim_option)
    print('3-2-3 ended')



    rid_list = ['NFiCk5XJ_OhZakVLcUSeUg', '8R8KCX3xRkMc7PfI7LeZbA',
       'D9gHjzqz4Vilxf-Hi04hgw', 'pQUWDMHkPf2II_uVDPCFFQ',
       'cBCMxmLdFoNqRsbWIYhnMw']
              
    # TODO - set algorithm for 3-3-1
    # KNNBasic, similarity : cosine
    sim_options = {'name': 'cosine', 'user_based': False}
    algo = surprise.KNNBasic(sim_options=sim_options)
    # TODO
    
    algo.fit(trainset)
    results = get_top_n(algo, testset, rid_list, n=5, user_based=False)
    with open('3-3-1.txt', 'w', encoding='utf-8') as f:
        for rid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Restaurant name %s top-5 results\n' % get_name_of_restaurant(rid))
            for uid, score in ratings:
                f.write('User ID %s\n\tscore %s\n' % (uid, str(score)))
            f.write('\n')
    print('3-3-1 ended')

    # TODO - set algorithm for 3-3-2
    # KNNWithMeans, similarity : pearson
    sim_options = {'name': 'pearson', 'user_based': False}
    algo = surprise.KNNWithMeans(sim_options=sim_options)
    # TODO
    
    algo.fit(trainset)
    results = get_top_n(algo, testset, rid_list, n=5, user_based=False)
    with open('3-3-2.txt', 'w', encoding='utf-8') as f:
        for rid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Restaurant name %s top-5 results\n' % get_name_of_restaurant(rid))
            for uid, score in ratings:
                f.write('User ID %s\n\tscore %s\n' % (uid, str(score)))
            f.write('\n')
    print('3-3-2 ended')


    # TODO - 3-3-3. Best Model

    algorithms = [surprise.KNNBasic, surprise.KNNWithMeans, surprise.KNNWithZScore, surprise.KNNBaseline]
    sim_options = [{'name': 'msd','user_based': False}, {'name': 'cosine','user_based': False}, {'name': 'pearson','user_based': False}, {'name': 'pearson_baseline','user_based': False}]
    np.random.seed(0)
    kf = KFold(n_splits=5)
    best_algo_ib = None
    best_sim_option_ib = None
    best_score_ib = float('inf')

    for algo in algorithms:
        for sim_option in sim_options:
            algo_instance = algo(sim_options=sim_option)
            rmse_scores = []
            for trainset, testset in kf.split(data):
                algo_instance.fit(trainset)
                predictions = algo_instance.test(testset)
                rmse = surprise.accuracy.rmse(predictions, verbose=True)
                rmse_scores.append(rmse)
            mean_rmse = np.mean(rmse_scores)
            if mean_rmse < best_score_ib:
                best_algo_ib = algo
                best_sim_option_ib = sim_option
                best_score_ib = mean_rmse

    # print the best algorithm, similarity option, and the average RMSE
    print(f"Best Algorithm: {best_algo_ib.__name__}")
    print(f"Best Similarity Option: {best_sim_option_ib}")
    print(f"Best Score (Average RMSE): {best_score_ib}")

    # best algorithm and, result : 
    ''' 
    Best Algorithm: KNNBasic
    Best Similarity Option: {'name': 'pearson', 'user_based': False}
    Best Score (Average RMSE): 0.932072132214174
    '''

    best_algo_ib = best_algo_ib(sim_options=best_sim_option_ib)

    print('3-3-3 ended')
    # TODO




    # TODO - set algorithm for 3-4-1
    # SVD, n_factors : 100, n_epochs : 50, biased : False
    algo = surprise.SVD(n_factors=100, n_epochs=50, biased=False)
    # TODO
    
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('3-4-1.txt', 'w', encoding='utf-8') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for rid, score in ratings:
                f.write('Restaurant name %s\n\tscore %s\n' % (get_name_of_restaurant(rid), str(score)))
            f.write('\n')
    print('3-4-1 ended')

    # TODO - set algorithm for 3-4-2
    # SVD, n_factors : 200, n_epochs : 100, biased : True
    algo = surprise.SVD(n_factors=200, n_epochs=100, biased=True)
    # TODO
    
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('3-4-2.txt', 'w', encoding='utf-8') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for rid, score in ratings:
                f.write('Restaurant name %s\n\tscore %s\n' % (get_name_of_restaurant(rid), str(score)))
            f.write('\n')
    print('3-4-2 ended')

    # TODO - set algorithm for 3-4-3
    # SVD++, n_factors : 100, n_epochs : 50
    algo = surprise.SVDpp(n_factors=100, n_epochs=50)
    # TODO
    
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('3-4-3.txt', 'w', encoding='utf-8') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for rid, score in ratings:
                f.write('Restaurant name %s\n\tscore %s\n' % (get_name_of_restaurant(rid), str(score)))
            f.write('\n')
    print('3-4-3 ended')

    # TODO - set algorithm for 3-4-4
    # SVD++, n_factors : 100, n_epochs : 100
    algo = surprise.SVDpp(n_factors=100, n_epochs=100)
    # TODO
    
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('3-4-4.txt', 'w', encoding='utf-8') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for rid, score in ratings:
                f.write('Restaurant name %s\n\tscore %s\n' % (get_name_of_restaurant(rid), str(score)))
            f.write('\n')
    print('3-4-4 ended')


    # TODO - 3-4-5. Best Model    
    # matrix based => SVD SVD++ NMF algorithms 

    # algorithm list
    algorithms = [surprise.SVD, surprise.SVDpp, surprise.NMF]

    # parameter grid
    param_grid_svd = {'n_factors': [100, 150, 200, 250, 300], 'n_epochs': [50, 100, 150, 200], 'biased': [True, False]}
    param_grid_svdpp = {'n_factors': [100, 150, 200, 250, 300], 'n_epochs': [50, 100, 150, 200]}
    param_grid_nmf = {'n_factors': [100, 150, 200, 250, 300], 'n_epochs': [50, 100, 150, 200]}

    best_score_mf = float('inf')
    best_algo_mf = None
    best_param = None

    # SVD grid search
    gs_svd = GridSearchCV(surprise.SVD, param_grid_svd, measures=['rmse'], cv=5)
    gs_svd.fit(data)
    if gs_svd.best_score['rmse'] < best_score_mf:
        best_algo_mf = surprise.SVD
        best_param = gs_svd.best_params['rmse']
        best_score_mf = gs_svd.best_score['rmse']

    print('SVD search ended')

    # SVDpp grid search
    gs_svdpp = GridSearchCV(surprise.SVDpp, param_grid_svdpp, measures=['rmse'], cv=5)
    gs_svdpp.fit(data)
    if gs_svdpp.best_score['rmse'] < best_score_mf:
        best_algo_mf = surprise.SVDpp
        best_param = gs_svdpp.best_params['rmse']
        best_score_mf = gs_svdpp.best_score['rmse']

    print('SVD++ search ended')

    # NMF grid search
    gs_nmf = GridSearchCV(surprise.NMF, param_grid_nmf, measures=['rmse'], cv=5)
    gs_nmf.fit(data)
    if gs_nmf.best_score['rmse'] < best_score_mf:
        best_algo_mf = surprise.NMF
        best_param = gs_nmf.best_params['rmse']
        best_score_mf = gs_nmf.best_score['rmse']

    print('NMF search ended')

    print(f"Best Algorithm: {best_algo_mf.__name__}")
    print(f"Best Parameters: {best_param}")
    print(f"Best Score (RMSE): {best_score_mf}")

    '''
    Best Algorithm: SVDpp
    Best Parameters: {'n_factors': 300, 'n_epochs': 50}
    Best Score (RMSE): 0.9885273757330861
    '''

    print('3-4-5 ended')


    best_algo_mf = best_algo_mf(**best_param)

        # TODO
            

#%%
if __name__ == '__main__':
    part1()
    part2()
    part3()



# %%
# part2-4 따로 실행하는 용도
# data frame and data analysis (part2_association.pkl)
df = pd.read_pickle(f'DMA_project2_team09_part2_association.pkl')
# Total number of association rules
print(f"Total number of association rules: {len(df)}")
# Top 10 rules sorted by lift
print("\nTop 10 rules sorted by lift:")
print(df.sort_values('lift', ascending=False).head(10))
# Quantitative Analysis
print("\nQuantitative Analysis:")
# Descriptive statistics of support, confidence, and lift
print(df[['support', 'confidence', 'lift']].describe())

# %%
