import dropbox
import os
import pathlib
import dropbox
import re


def upload_file_to_dropbox(filepath_to_upload, dropbox_filepath, dropbox_token=None):
    if dropbox_token == None:
        dropbox_token = os.environ["DROPBOX_TOKEN"]

    dbox_api = dropbox.Dropbox(your_api_access_token)

    # open the file and upload it
    with open(filepath_to_upload, "rb") as f:
        # upload gives you metadata about the file
        # we want to overwite any previous version of the file
        meta = dbox_api.files_upload(
            f.read(), dropbox_filepath, mode=dropbox.files.WriteMode("overwrite")
        )

    # create a shared link
    link = dbox_api.sharing_create_shared_link(targetfile)

    # link which directly downloads by replacing ?dl=0 with ?dl=1
    dl_url = re.sub(r"\?dl\=0", "?dl=1", link.url)

    return dl_url


if __name__ == "__main__":
    your_api_access_token = os.environ["DROPBOX_TOKEN"]

    # the source file
    folder = pathlib.Path(".")  # located in this folder
    filename = "test.txt"  # file name
    filepath = folder / filename  # path object, defining the file

    # target location in Dropbox
    target = "/temp2/"  # the target folder
    targetfile = target + filename  # the target path and file name
    dl_url = upload_file_to_dropbox(
        filepath, targetfile, dropbox_token=your_api_access_token
    )
    print(dl_url)
