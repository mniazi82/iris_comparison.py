from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. بارگذاری داده‌ها
iris = load_iris()
X = iris.data
y = iris.target

# 2. تقسیم داده‌ها به دو گروه آموزش (Train) و تست (Test)
# test_size=0.3 یعنی ۳۰ درصد داده‌ها برای تست و ۷۰ درصد برای آموزش
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------
# روش اول: Gaussian Naive Bayes (روش خواسته شده در تمرین)
gnb = GaussianNB()
gnb.fit(X_train, y_train)           # آموزش مدل
y_pred_gnb = gnb.predict(X_test)    # پیش‌بینی روی داده‌های تست
acc_gnb = accuracy_score(y_test, y_pred_gnb) # محاسبه دقت

# ---------------------------------------------------------
# روش دوم: K-Nearest Neighbors (برای مقایسه)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)           # آموزش مدل
y_pred_knn = knn.predict(X_test)    # پیش‌بینی روی داده‌های تست
acc_knn = accuracy_score(y_test, y_pred_knn) # محاسبه دقت

# ---------------------------------------------------------
# چاپ و مقایسه نتایج
print(f"Accuracy of GaussianNB: {acc_gnb * 100:.2f}%")
print(f"Accuracy of KNN (k=3):  {acc_knn * 100:.2f}%")

if acc_gnb > acc_knn:
    print("GaussianNB performed better.")
elif acc_knn > acc_gnb:
    print("KNN performed better.")
else:
    print("Both models have the same accuracy.")  
