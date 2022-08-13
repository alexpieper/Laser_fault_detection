from data_preprocessing import DataPreprocessing
from data_modelling import NaiveModel, LinearModel, NeuralNet, XGBoost, DecisionTree, RandomForest, NeuralNetSklearn, ExplainableClassifier
from result_analysis import ModelAnalysis
import dashboard


def main():
    ###############################
    ######## Data Analysis ########
    ###############################

    preprocessing = DataPreprocessing()
    preprocessing.get_stats_of_data()
    preprocessing.visualize_raw_data(n_obs=7)
    
    
    #############################################################
    ######## Data Preprocessing and Train/Test Splitting ########
    #############################################################
    
    training_ratio = 0.8
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = preprocessing.get_train_test_sets(train_ratio=training_ratio,
                                                                                         normalizing=False,
                                                                                         feature_engineering=False)
    X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = preprocessing.get_train_test_sets(train_ratio=training_ratio,
                                                                                                                     normalizing=True,
                                                                                                                     feature_engineering=False)
    X_train_feat_eng_raw, X_test_feat_eng_raw, y_train_feat_eng_raw, y_test_feat_eng_raw = preprocessing.get_train_test_sets(train_ratio=training_ratio,
                                                                                                                             normalizing=False,
                                                                                                                             feature_engineering=True,
                                                                                                                             forward_diff=5)
    X_train_feat_eng_normalized, X_test_feat_eng_normalized, y_train_feat_eng_normalized, y_test_feat_eng_normalized = preprocessing.get_train_test_sets(train_ratio=training_ratio,
                                                                                                                                                        normalizing=True,
                                                                                                                                                        feature_engineering=True,
                                                                                                                                                        forward_diff=5)






    ###############################
    ######## Model Creation #######
    ###############################

    # Data Modelling for Naive model
    naive_model = NaiveModel(X_train_raw, y_train_raw, 8, 'Naive Model')
    naive_model.train_model()







    # Data Modelling for Naive model
    decision_tree_model_raw = DecisionTree(X_train_raw, y_train_raw, 'Decision Tree (raw)')
    decision_tree_model_raw.train_model()

    # Data Modelling for Linear model
    decision_tree_model_norm = DecisionTree(X_train_normalized, y_train_normalized, 'Decision Tree (normalized)')
    decision_tree_model_norm.train_model()

    # Data Modelling for Linear model
    decision_tree_model_feat = DecisionTree(X_train_feat_eng_raw, y_train_feat_eng_raw, 'Decision Tree (feature engineered)')
    decision_tree_model_feat.train_model()

    # Data Modelling for Linear model
    decision_tree_model_norm_feat = DecisionTree(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'Decision Tree (normalized, feature engineered)')
    decision_tree_model_norm_feat.train_model()










    # Data Modelling for Naive model
    random_forest_model_raw = RandomForest(X_train_raw, y_train_raw, 'Random Forest (raw)')
    random_forest_model_raw.train_model()

    # Data Modelling for Linear model
    random_forest_model_norm = RandomForest(X_train_normalized, y_train_normalized, 'Random Forest (normalized)')
    random_forest_model_norm.train_model()

    # Data Modelling for Linear model
    random_forest_model_feat = RandomForest(X_train_feat_eng_raw, y_train_feat_eng_raw, 'Random Forest (feature engineered)')
    random_forest_model_feat.train_model()

    # Data Modelling for Linear model
    random_forest_model_norm_feat = RandomForest(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'Random Forest (normalized, feature engineered)')
    random_forest_model_norm_feat.train_model()







    # Data Modelling for Linear model
    explain_ml_raw = ExplainableClassifier(X_train_raw, y_train_raw, 'Explainable Classifier (raw)')
    explain_ml_raw.train_model()
    
    # Data Modelling for Linear model
    explain_ml_norm = ExplainableClassifier(X_train_normalized, y_train_normalized, 'Explainable Classifier (normalized)')
    explain_ml_norm.train_model()

    # Data Modelling for Linear model
    explain_ml_feat = ExplainableClassifier(X_train_feat_eng_raw, y_train_feat_eng_raw, 'Explainable Classifier (feature engineered)')
    explain_ml_feat.train_model()

    # Data Modelling for Linear model
    explain_ml_norm_feat = ExplainableClassifier(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'Explainable Classifier (normalized, feature engineered)')
    explain_ml_norm_feat.train_model()

    # explainable_model_raw.explain_model()




    # Data Modelling for Linear model
    linear_model_raw = LinearModel(X_train_raw, y_train_raw, 'Linear Model (raw)')
    linear_model_raw.train_model()

    # Data Modelling for Linear model
    linear_model_norm = LinearModel(X_train_normalized, y_train_normalized, 'Linear Model (normalized)')
    linear_model_norm.train_model()

    # Data Modelling for Linear model
    linear_model_feat = LinearModel(X_train_feat_eng_raw, y_train_feat_eng_raw, 'Linear Model (feature engineered)')
    linear_model_feat.train_model()

    # Data Modelling for Linear model
    linear_model_norm_feat = LinearModel(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'Linear Model (normalized, feature engineered)')
    linear_model_norm_feat.train_model()









    # Data Modelling for Linear model
    xgb_model_raw = XGBoost(X_train_raw, y_train_raw, 'XGB Model (raw)')
    xgb_model_raw.train_model()

    # Data Modelling for Linear model
    xgb_model_norm = XGBoost(X_train_normalized, y_train_normalized, 'XGB Model (normalized)')
    xgb_model_norm.train_model()

    xgb_model_feat = XGBoost(X_train_feat_eng_raw, y_train_feat_eng_raw, 'XGB Model (feature engineered)')
    xgb_model_feat.train_model()

    # Data Modelling for Linear model
    xgb_model_norm_feat = XGBoost(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'XGB Model (normalized, feature engineered)')
    xgb_model_norm_feat.train_model()











    learning_rate = 0.01
    epochs = 5

    # Data Modelling for Linear model
    ann_model_raw = NeuralNet(X_train_raw, y_train_raw, learning_rate , epochs,'ANN 1 (raw)')
    ann_model_raw.train_model()

    # Data Modelling for Linear model
    ann_model_norm = NeuralNet(X_train_normalized, y_train_normalized, learning_rate , epochs,'ANN 1 (normalized)')
    ann_model_norm.train_model()

    # Data Modelling for Linear model
    ann_model_feat = NeuralNet(X_train_feat_eng_raw, y_train_feat_eng_raw, learning_rate , epochs,'ANN 1 (feature engineered)')
    ann_model_feat.train_model()

    # Data Modelling for Linear model
    ann_model_norm_feat = NeuralNet(X_train_feat_eng_normalized, y_train_feat_eng_normalized, learning_rate , epochs,'ANN 1 (normalized, feature engineered)')
    ann_model_norm_feat.train_model()




    learning_rate = 0.01
    epochs = 60

    # Data Modelling for Linear model
    ann_model_2_raw = NeuralNet(X_train_raw, y_train_raw, learning_rate , epochs,'ANN 2 (raw)')
    ann_model_2_raw.train_model()

    # Data Modelling for Linear model
    ann_model_2_norm = NeuralNet(X_train_normalized, y_train_normalized, learning_rate , epochs,'ANN 2 (normalized)')
    ann_model_2_norm.train_model()

    # Data Modelling for Linear model
    ann_model_2_feat = NeuralNet(X_train_feat_eng_raw, y_train_feat_eng_raw, learning_rate , epochs,'ANN 2 (feature engineered)')
    ann_model_2_feat.train_model()

    # Data Modelling for Linear model
    ann_model_2_norm_feat = NeuralNet(X_train_feat_eng_normalized, y_train_feat_eng_normalized, learning_rate , epochs,'ANN 2 (normalized, feature engineered)')
    ann_model_2_norm_feat.train_model()



    # Data Modelling for Linear model
    ann_model_sklearn_raw = NeuralNetSklearn(X_train_raw, y_train_raw, 'ANN Sklearn (raw)')
    ann_model_sklearn_raw.train_model()

    # Data Modelling for Linear model
    ann_model_sklearn_norm = NeuralNetSklearn(X_train_normalized, y_train_normalized, 'ANN Sklearn (normalized)')
    ann_model_sklearn_norm.train_model()

    # Data Modelling for Linear model
    ann_model_sklearn_feat = NeuralNetSklearn(X_train_feat_eng_raw, y_train_feat_eng_raw, 'ANN Sklearn (feature engineered)')
    ann_model_sklearn_feat.train_model()

    # Data Modelling for Linear model
    ann_model_sklearn_norm_feat = NeuralNetSklearn(X_train_feat_eng_normalized, y_train_feat_eng_normalized, 'ANN Sklearn (normalized, feature engineered)')
    ann_model_sklearn_norm_feat.train_model()





    ###############################
    ######## Model Evaluation #####
    ###############################
    ModelAnalysis(X_test_raw, y_test_raw, naive_model).make_whole_report()

    ModelAnalysis(X_test_raw, y_test_raw, linear_model_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, linear_model_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, linear_model_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, linear_model_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, decision_tree_model_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, decision_tree_model_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, decision_tree_model_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, decision_tree_model_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, random_forest_model_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, random_forest_model_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, random_forest_model_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, random_forest_model_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, explain_ml_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, explain_ml_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, explain_ml_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, explain_ml_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, xgb_model_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, xgb_model_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, xgb_model_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, xgb_model_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, ann_model_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, ann_model_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, ann_model_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, ann_model_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, ann_model_2_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, ann_model_2_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, ann_model_2_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, ann_model_2_norm_feat).make_whole_report()


    ModelAnalysis(X_test_raw, y_test_raw, ann_model_sklearn_raw).make_whole_report()
    ModelAnalysis(X_test_normalized, y_test_normalized, ann_model_sklearn_norm).make_whole_report()
    ModelAnalysis(X_test_feat_eng_raw, y_test_feat_eng_raw, ann_model_sklearn_feat).make_whole_report()
    ModelAnalysis(X_test_feat_eng_normalized, y_test_feat_eng_normalized, ann_model_sklearn_norm_feat).make_whole_report()



    

    ###############################################
    ######## Interactive Dashboard Creation #######
    ###############################################

    dashboard.make_evaluation_dashboard(X_test_raw,
                                        y_test_raw,
                                        naive_model,
                                        linear_model_raw,
                                        linear_model_norm,
                                        linear_model_feat,
                                        linear_model_norm_feat,
                                        decision_tree_model_raw,
                                        decision_tree_model_norm,
                                        decision_tree_model_feat,
                                        decision_tree_model_norm_feat,
                                        random_forest_model_raw,
                                        random_forest_model_norm,
                                        random_forest_model_feat,
                                        random_forest_model_norm_feat,
                                        xgb_model_raw,
                                        xgb_model_norm,
                                        xgb_model_feat,
                                        xgb_model_norm_feat,
                                        explain_ml_raw,
                                        explain_ml_norm,
                                        explain_ml_feat,
                                        explain_ml_norm_feat,
                                        ann_model_raw,
                                        ann_model_norm,
                                        ann_model_feat,
                                        ann_model_norm_feat,
                                        ann_model_2_raw,
                                        ann_model_2_norm,
                                        ann_model_2_feat,
                                        ann_model_2_norm_feat,
                                        ann_model_sklearn_raw,
                                        ann_model_sklearn_norm,
                                        ann_model_sklearn_feat,
                                        ann_model_sklearn_norm_feat,
                                        )

if __name__ == '__main__':
    main()

