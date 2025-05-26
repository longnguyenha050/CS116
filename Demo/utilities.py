def numerical_rating(userRating):
    if userRating == "Very Poor":
        return 1
    elif userRating == "Poor":
        return 2
    elif userRating == "Fair":
        return 3
    elif userRating == "Below Average":
        return 4
    elif userRating == "Average" or userRating == "Cannot Say":
        return 5
    elif userRating == "Above Average":
        return 6
    elif userRating == "Good":
        return 7
    elif userRating == "Very Good":
        return 8
    elif userRating == "Excellent":
        return 9
    elif userRating == "Very Excellent":
        return 10