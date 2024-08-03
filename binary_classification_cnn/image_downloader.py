from pygoogle_image import image as pi

'''Pygoogle Image is a Python library designed to simplify the process of extracting images from Google Image Search. With just a keyword, you can download as many images as you need.'''


name = input("Enter keyword : ")
name1 = str(f'{name}')
numb = int(input('Enter how many images you want : '))
pi.download(keywords= name1,limit= numb)
