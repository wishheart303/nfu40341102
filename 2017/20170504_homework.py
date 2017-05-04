from instagram.client import InstagramAPI
access_token = "1308465562.eb196eb.a577b22057674f2b89126ccf34b040fd"
client_secret = "4142fd81a5c44b5d8ba8aaabbaa6e9c4"

api = InstagramAPI(access_token=access_token, client_secret=client_secret,)
user_id = api.user_search('ying_789')[0].id

recent_media, next_ = api.user_recent_media(user_id=user_id, count=5)

for media in recent_media:
    print (media.caption.text)
    print ('<img src="%s"/>' % media.images['thumbnail'].url)