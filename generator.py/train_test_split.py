from sklearn.model_selection import train_test_split

def train_test_split_data(df,target_col,train_ratio=0.8):
    x=df.drop(target_col,axis=1)
    y=df[target_col]
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1-train_ratio,random_state=42)
    return x_train,x_test,y_train,y_test
    