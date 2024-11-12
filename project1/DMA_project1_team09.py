import mysql.connector
import csv
# TODO: REPLACE THE VALUE OF VARIABLE team (EX. TEAM 1 --> team = 1)
team = 9

# Requirement1: create schema ( name: DMA_team## )
def requirement1(host, user, password):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    print('Creating schema...')
    
    # TODO: WRITE CODE HERE
    # Check if the schema already exists
    cursor.execute("SHOW DATABASES LIKE 'DMA_team09'")
    # LIKE keyword => Find "DMA_team09" database
    # SHOW DATABASES : Show all databases in the server

    result = cursor.fetchall()
    #fetchall() : fetch all rows of a query result, returning them as a list of tuples.
    # If no more rows are available, it returns an empty list.

    if result:
        print('Schema already exists')
    else:
        cursor.execute('CREATE DATABASE IF NOT EXISTS DMA_team09;')
        print('Schema created')

    # TODO: WRITE CODE HERE
    
    cursor.close()

# Requierement2: create table
def requirement2(host, user, password):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    print('Creating tables...')
    
    # TODO: WRITE CODE HERE
    cursor.execute('USE DMA_team09')

    
    # 1. User table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS User(
    user_id VARCHAR(255) NOT NULL,
    user_name VARCHAR(255) NOT NULL,
    region INT(11),
    primary key(user_id));
    ''')



    # 2. restaurant table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Restaurant(
    restaurant_id VARCHAR(255) NOT NULL,
    restaurant_name VARCHAR(255) NOT NULL,
    lunch_price_min INT(11),
    lunch_price_max INT(11),
    dinner_price_min INT(11),
    dinner_price_max INT(11),
    location INT(11) NOT NULL,
    category INT(11) NOT NULL,                     
    primary key(restaurant_id));
    ''')

    # 3. review table creation

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Review(
    review_id INT(11) NOT NULL,
    review_content VARCHAR(255),
    reg_date DATETIME,
    user_id VARCHAR(255) NOT NULL,
    total_score DECIMAL(11) NOT NULL,
    taste_score DECIMAL(11),
    service_score DECIMAL(11),
    mood_score DECIMAL(11),
    restaurant VARCHAR(255) NOT NULL,
    primary key(review_id));
    ''')

    # 4. Menu table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Menu(
    menu_name VARCHAR(255) NOT NULL,
    price_min INT(11),
    price_max INT(11),
    restaurant VARCHAR(255) NOT NULL,
    primary key(restaurant, menu_name));
    ''')    

    # 5. Post_Menu table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Post_Menu(
    post_id INT(11) NOT NULL,
    menu_name VARCHAR(255) NOT NULL,
    restaurant VARCHAR(255) NOT NULL,
    primary key(post_id,restaurant,menu_name));               
    ''')

    # 6. Location table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Location(
    name VARCHAR(255) NOT NULL,
    location_id INT(11) NOT NULL,
    primary key(location_id));               
    ''')

    # 7. Post table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Post(
    blog_title VARCHAR(255),
    blog_URL VARCHAR(255) NOT NULL,
    post_date DATETIME NOT NULL,
    restaurant VARCHAR(255) NOT NULL,
    post_id INT(11) NOT NULL,
    primary key(post_id));               
    ''')

    # 8. Follow table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Follow(
    followee_id VARCHAR(255) NOT NULL,
    follower_id VARCHAR(255) NOT NULL,
    primary key(followee_id, follower_id));               
    ''')

    # 9. Collection table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Collection(
    user_id VARCHAR(255) NOT NULL,
    restaurant_id VARCHAR(255) NOT NULL,
    primary key(user_id, restaurant_id));               
    ''')
    
    # 10. Category table creation
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Category(
    name VARCHAR(255) NOT NULL,
    category_id INT(11) NOT NULL,
    primary key(category_id));               
    ''')


    # TODO: WRITE CODE HERE
    cursor.close()




# Requirement3: insert data
def requirement3(host, user, password, directory):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    print('Inserting data...')
    
    # TODO: WRITE CODE HERE
    cursor.execute('USE DMA_team09')

    # 1. User table insertion
    filepath1 = directory + '/User.csv'
    with open(filepath1, 'r', encoding='utf-8') as User_data:
        for row1 in User_data.readlines()[1:]: 
            # create readlines() with [1:2] to output only as many rows as we want to try / test similarly for smaller files
            # When running as it is, a lot of data is inserted and it takes time, 
            # set something else to [1:2] when testing code, and run it with full data afterward
            row1 = row1.strip().split(',')
            for idx, data in enumerate(row1):
                if data == '':
                    row1[idx] = 'null'
                # convert string to int if possible
                try:
                    row1[idx] = int(data)
                except ValueError:
                    pass
            row1 = tuple(row1)
            sql1 = 'INSERT IGNORE INTO User VALUES {}'.format(row1)
  
           # Reason for IGNORE: If a record doesn't exist, insert it. If it exists, ignore it.
            # To prevent errors caused by duplicate insertions during iterations while testing code.

            sql1 = sql1.replace('\'null\'', 'null')
            cursor.execute(sql1)
    cnx.commit()

    print("User tuple insertion is done")


    # 2. Restaurant table insertion
    filepath2 = directory + '/Restaurant.csv'
    with open(filepath2, 'r', encoding='utf-8') as Restaurant_data:
        for row2 in Restaurant_data.readlines()[1:]: 
            row2 = row2.strip().split(',')
            for idx, data in enumerate(row2):
                if data == '':
                    row2[idx] = 'null'
                # convert string to int if possible
                try:
                    row2[idx] = int(data)
                except ValueError:
                    pass
            row2 = tuple(row2)

            sql2 = 'INSERT IGNORE INTO Restaurant VALUES {}'.format(row2)
            sql2 = sql2.replace('\'null\'', 'null')
            #print(sql2)
            cursor.execute(sql2)
    cnx.commit()
    
    print("Restaurant tuple insertion is done")

    # 3. Review table insertion
    filepath3 = directory + '/Review.csv'
    with open(filepath3, 'r', encoding='utf-8') as Review_data:
        for row3 in Review_data.readlines()[1:]: 
            row3 = row3.strip().split(',')
            for idx, data in enumerate(row3):
                if data == '':
                    row3[idx] = 'null'
                else:
                # convert string to int if possible
                # datetime type => str type
                    if idx == 0:
                        try:
                            row3[idx] = int(data)
                        except ValueError:
                            row3[idx] = 'null'
                    elif 4 <= idx <= 7:
                        try:
                            row3[idx] = float(data)
                        except ValueError:
                            row3[idx] = 'null'

            row3 = tuple(row3)

            sql3 = 'INSERT IGNORE INTO Review VALUES {}'.format(row3)
            sql3 = sql3.replace('\'null\'', 'null')
            #print(sql3)
            cursor.execute(sql3)
    cnx.commit()
    
    print("Review tuple insertion is done")
    
    # 4. Menu table insertion
    filepath4 = directory + '/Menu.csv'
    with open(filepath4, 'r', encoding='utf-8') as Menu_data:
        for row4 in Menu_data.readlines()[1:]: 
            row4 = row4.strip().split(',')
            for idx, data in enumerate(row4):
                if data == '':
                    row4[idx] = 'null'
                # convert string to int if possible
                try:
                    row4[idx] = int(float(data))
                    # I need to insert as INT, but the csv has prices in decimal, so I convert to float and then int
                except ValueError:
                    pass
            row4 = tuple(row4)

            sql4 = 'INSERT IGNORE INTO Menu VALUES {}'.format(row4)
            sql4 = sql4.replace('\'null\'', 'null')
            #print(sql4)
            cursor.execute(sql4)
    cnx.commit()

    print("Menu tuple insertion is done")

    # 5. Post_Menu table insertion
    filepath5 = directory + '/Post_Menu.csv'
    with open(filepath5, 'r', encoding='utf-8') as Post_Menu_data:
        for row5 in Post_Menu_data.readlines()[1:]: 
            row5 = row5.strip().split(',')
            for idx, data in enumerate(row5):
                if data == '':
                    row5[idx] = 'null'
                # convert string to int if possible
                try:
                    row5[idx] = int(float(data))
                except ValueError:
                    pass
            row5 = tuple(row5)

            sql5 = 'INSERT IGNORE INTO Post_Menu VALUES {}'.format(row5)
            sql5 = sql5.replace('\'null\'', 'null')
            #print(sql5)
            cursor.execute(sql5)
    cnx.commit()

    print("Post_Menu tuple insertion is done")
    
    # 6. Location table insertion
    filepath6 = directory + '/Location.csv'
    with open(filepath6, 'r', encoding='utf-8') as Location_data:
        for row6 in Location_data.readlines()[1:]: 
            row6 = row6.strip().split(',')
            for idx, data in enumerate(row6):
                if data == '':
                    row6[idx] = 'null'
                # convert string to int if possible
                try:
                    row6[idx] = int(data)
                except ValueError:
                    pass
            row6 = tuple(row6)

            sql6 = 'INSERT IGNORE INTO Location VALUES {}'.format(row6)
            sql6 = sql6.replace('\'null\'', 'null')
            #print(sql6)
            cursor.execute(sql6)
    cnx.commit()

    print("Location tuple insertion is done")
    
    # 7. Post table insertion
    filepath7 = directory + '/Post.csv'
    with open(filepath7, 'r', encoding='utf-8') as Post_data:
        for row7 in Post_data.readlines()[1:]: 
            row7 = row7.strip().split(',')
            for idx, data in enumerate(row7):
                if data == '':
                    row7[idx] = 'null'
                # convert string to int if possible
                try:
                    row7[idx] = int(data)
                except ValueError:
                    pass
            row7 = tuple(row7)

            sql7 = 'INSERT IGNORE INTO Post VALUES {}'.format(row7)
            sql7 = sql7.replace('\'null\'', 'null')
            #print(sql7)
            cursor.execute(sql7)
    cnx.commit()

    print("Post tuple insertion is done")
    
    # 8. Follow table insertion`
    filepath8 = directory + '/Follow.csv'
    with open(filepath8, 'r', encoding='utf-8') as Follow_data:
        for row8 in Follow_data.readlines()[1:]: 
            row8 = row8.strip().split(',')
            for idx, data in enumerate(row8):
                if data == '':
                    row8[idx] = 'null'
            row8 = tuple(row8)

            sql8 = 'INSERT IGNORE INTO Follow VALUES {}'.format(row8)
            sql8 = sql8.replace('\'null\'', 'null')
            #print(sql8)
            cursor.execute(sql8)    
    cnx.commit()

    print("Follow tuple insertion is done")

    # 9. Collection table insertion
    filepath9 = directory + '/Collection.csv'
    with open(filepath9, 'r', encoding='utf-8') as Collection_data:
        for row9 in Collection_data.readlines()[1:]: 
            row9 = row9.strip().split(',')
            for idx, data in enumerate(row9):
                if data == '':
                    row9[idx] = 'null'
            row9 = tuple(row9)

            sql9 = 'INSERT IGNORE INTO Collection VALUES {}'.format(row9)
            sql9 = sql9.replace('\'null\'', 'null')
            #print(sql9)
            cursor.execute(sql9)
    cnx.commit()

    print("Collection tuple insertion is done")

    # 10. Category table insertion
    filepath10 = directory + '/Category.csv'
    with open(filepath10, 'r', encoding='utf-8') as Category_data:
        for row10 in Category_data.readlines()[1:]: 
            row10 = row10.strip().split(',')
            for idx, data in enumerate(row10):
                if data == '':
                    row10[idx] = 'null'
                # convert string to int if possible
                try:
                    row10[idx] = int(data)
                except ValueError:
                    pass
            row10 = tuple(row10)

            sql10 = 'INSERT IGNORE INTO Category VALUES {}'.format(row10)
            sql10 = sql10.replace('\'null\'', 'null')
            #print(sql10)
            cursor.execute(sql10)
        cnx.commit()
    

    print("Category tuple insertion is done")

    # TODO: WRITE CODE HERE
    cursor.close()


# Requirement4: add constraint (foreign key)
def requirement4(host, user, password):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    print('Adding constraints...')
    cursor.execute('USE DMA_team09')
    # TODO: WRITE CODE HERE
    # 1. User's foreign key
    cursor.execute("ALTER TABLE User ADD CONSTRAINT FOREIGN KEY (region) REFERENCES Location(location_id)")
    print("1-1 done")

    # 2. Restaurant's foreign key
    cursor.execute("ALTER TABLE Restaurant ADD CONSTRAINT FOREIGN KEY (location) REFERENCES Location(location_id)")
    print("2-1 done")
    cursor.execute("ALTER TABLE Restaurant ADD CONSTRAINT FOREIGN KEY (category) REFERENCES Category(category_id)")
    print("2-2 done")
    
    # 3. Review's foreign key
    cursor.execute("ALTER TABLE Review ADD CONSTRAINT FOREIGN KEY (user_id) REFERENCES User(user_id)")
    print("3-1 done")
    cursor.execute("ALTER TABLE Review ADD CONSTRAINT FOREIGN KEY (restaurant) REFERENCES Restaurant(restaurant_id)")
    print("3-2 done")
    
    # 4. Post's foreign key
    cursor.execute("ALTER TABLE Post ADD CONSTRAINT FOREIGN KEY (restaurant) REFERENCES Restaurant(restaurant_id)")
    print("4-1 done")

    # 5. Menu's foreign key
    cursor.execute("ALTER TABLE Menu ADD CONSTRAINT FOREIGN KEY (restaurant) REFERENCES Restaurant(restaurant_id)")
    print("5-1 done")

    # 6. Collection's foreign key
    cursor.execute("ALTER TABLE Collection ADD CONSTRAINT FOREIGN KEY (restaurant_id) REFERENCES Restaurant(restaurant_id)")
    print("6-1 done")
    cursor.execute("ALTER TABLE Collection ADD CONSTRAINT FOREIGN KEY (user_id) REFERENCES User(user_id)")
    print("6-2 done")

    # 7. Post_Menu's foreign key
    cursor.execute("ALTER TABLE Post_Menu ADD CONSTRAINT FOREIGN KEY (post_id) REFERENCES Post(post_id)")
    print("7-1 done")
    cursor.execute("ALTER TABLE Post_Menu ADD CONSTRAINT FOREIGN KEY (restaurant, menu_name) REFERENCES Menu(restaurant, menu_name)")
    print("7-2 done")

    # 8. Follow's foreign key
    cursor.execute("ALTER TABLE Follow ADD CONSTRAINT FOREIGN KEY (followee_id) REFERENCES User(user_id)")
    print("8-1 done")
    cursor.execute("ALTER TABLE Follow ADD CONSTRAINT FOREIGN KEY (follower_id) REFERENCES User(user_id)")

    print("8-2 done")



    # TODO: WRITE CODE HERE
    cursor.close()
    

# TODO: REPLACE THE VALUES OF FOLLOWING VARIABLES
host = 'localhost'
user = 'root'
password = 'password'
directory_in = './dataset'


requirement1(host=host, user=user, password=password)
requirement2(host=host, user=user, password=password)
requirement3(host=host, user=user, password=password, directory=directory_in)
requirement4(host=host, user=user, password=password)
print('Done!')