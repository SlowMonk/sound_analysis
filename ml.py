
# Training
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from utils import print_scores
from xgboost import XGBClassifier

def train_svm(C, x_train_scaled, y_train, x_test_pcs, y_test):

    print('############################ svm train ############################')
    clf = svm.SVC(C=C, kernel="linear",verbose=False)  # kernel methods: linear, polynomial, sigmoid, rbf
    clf.fit(x_train_scaled, y_train)

    # Predict the test dataset using the trained SVM
    y_pred = clf.predict(x_test_pcs)
    print_scores(y_test, y_pred)

def train_xgboost(x_train_scaled, y_train, x_test_pcs, y_test):
    print('############################ xgboost train ############################')
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)  # 1000개의 가지? epoch? , 0.05 학습률
    xgb.fit(x_train_scaled, y_train)  # 학습

    y_pred = xgb.predict(x_test_pcs)  # 검증

    # # Decode the predicted labels
    # y_pred = encoder.inverse_transform(y_pred_xgb)
    print_scores(y_test, y_pred)
