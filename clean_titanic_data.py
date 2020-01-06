import feature_engineering as fe
from sklearn.model_selection import train_test_split


def clean_titanic_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    del x, y

    # Extract length and title of name
    print("Prepare data")
    x_train, x_test = fe.extractFromNames(x_train, x_test)

    # Replace missing age values with means grouped  by title and class
    x_train, x_test = fe.imputeAge(x_train, x_test)

    # Take first letter of cabin
    x_train, x_test = fe.extractCabinLetter(x_train, x_test)

    # Fill missing values for embarked
    x_train, x_test = fe.imputeEmbarked(x_train, x_test)

    # Determine family size
    x_train, x_test = fe.extractFamilySize(x_train, x_test)

    # Extract ticket length
    x_train['Ticket_Len'] = x_train['Ticket'].apply(lambda x: len(x))
    x_test['Ticket_Len'] = x_test['Ticket'].apply(lambda x: len(x))

    # Create dummy variable for several columns
    x_train, x_test = fe.createDummies(x_train, x_test,
                                       columns=['Pclass', 'Sex', 'Embarked', 'Cabin_Letter', 'Name_Title', 'Fam_Size'])

    # Delete unused columns
    x_train = x_train.drop(columns="Ticket")
    x_test = x_test.drop(columns="Ticket")

    return x_train, x_test, y_train, y_test
