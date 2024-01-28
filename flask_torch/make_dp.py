# 데이터베이스 생성 및 테이블 생성
import sqlite3

conn = sqlite3.connect('sample_data.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM images ")
result = cursor.fetchall()
import pdb;pdb.set_trace()
# # # 예제 데이터 추가
# cursor.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label INTEGER)''')
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image0.jpg', 0)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image1.jpg', 1)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image2.jpg', 2)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image3.jpg', 3)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image4.jpg', 4)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image5.jpg', 5)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image6.jpg', 6)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image7.jpg', 7)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image9.jpg', 8)")
# cursor.execute("INSERT INTO images (filename, label) VALUES ('image9.jpg', 8)")
# ... 추가 데이터 추가 ...

conn.commit()
conn.close()
