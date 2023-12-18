from model import NgramModel, NgramModelWithInterpolation,create_ngram_model


def main():

    flag = None
    try:
        flag = int(input("Do you want to use interpolation\n 1) Yes\n 2) No\n"))
        if flag != 1 and flag != 2:
            flag = 2
            raise Exception("Invalid option selected.")
    except:
        print("You entered something wrong, you'll use the regular model")


    context_size = int(input("Enter the context size (n) for the n-gram model: "))

    m = None
    if flag == 1:
        print("Creating Interpolated Model")
        print("Inputting Shakespeare Text")
        m = create_ngram_model(NgramModelWithInterpolation, './shakespeare_input.txt', context_size)
    else:
        print("Creating model without interpolation")
        print("Inputting Shakespeare Text")
        m = create_ngram_model(NgramModel, './shakespeare_input.txt', context_size)


    # Generate some random text
    print("\nGenerated text:\n\n")
    print(m.random_text(600))


if __name__ == "__main__":
    main()
