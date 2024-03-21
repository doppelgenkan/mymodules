import numpy as np
import pandas as pd
import random
import sys


def studcsv(fn, addfn):
    org = '/takarazuka'
    if addfn == None:
        addfn = 'users'
    df = pd.read_csv(fn, header=None, skiprows=1)
    email_lis = []
    pw_lis = []
    org_lis = []
    id_lis = []
    dept_lis =[]
    name_arr = np.array([k.split('　') for k in (df.iloc[:,1])]).T

    
    for id_num in df.iloc[:,0]:

        # ----- 学科識別とユーザーネーム上2文字の置き換え -----
        if id_num[2:4] == ('1P' or '1Ｐ'):
            top = 'pt'
            dept = '理学療法'
        elif id_num[2:4] == ('1A' or '1Ａ'):
            top = 'am'
            dept = '鍼灸'
        elif id_num[2:4] == ('1J' or '1Ｊ'):
            top = 'jt'
            dept = '柔道整復'
        elif id_num[2:4] == ('1D' or '1Ｄ'):
            top = 'dh'
            dept = '口腔保健'
        elif id_num[2:4] == ('SW' or 'ＳＷ'):
            top = 'sw'
            dept = '社会福祉養成課程'
        else:
            top = 'unknown'
            dept = '存在しない'

        #print(id_num[2:4])
        #print(top)

        # ----- Emailアドレス生成とそのリスト -----
        email = top + id_num[:2] + id_num[4:] + '@stud.tumh.ac.jp'
        email_lis.append(email)
        
        # ----- パスワード生成とそのリスト -----
        pw = chr(65+random.randint(0,25))
        for j in range(7):
            pw += chr(65+32+random.randint(0,25))
        for k in range(4):
            pw += chr(48+random.randint(0,9))
        pw_lis.append(pw)

        # ----- 学籍番号リスト -----
        id_lis.append(id_num)

        # ----- 組織識別子リスト -----
        org_lis.append(f'{org}/{top}')
        
        # ----- 学科リスト -----
        dept_lis.append(dept)

    # ----- アップロードCSV用データフレーム生成 -----
    mk_df = pd.DataFrame(
        {'Last Name [Required]':name_arr[0],
         'First Name [Required]':name_arr[1],
         'Email Address [Required]':email_lis,
         'Password [Required]':pw_lis,
         'Org Unit Path [Required]':org_lis,
         'Employee ID':id_lis,
         'Department':dept_lis})

    # ----- アップロード用CSVファイル生成 -----
    mk_df.to_csv(f'/Users/me/Downloads/{addfn}', index=False)


if __name__ == '__main__':
    try:
        fn = sys.argv[1]
        if len(sys.argv) < 3:
            addfn = 'users.csv'
        else:
            addfn = sys.argv[2]
        studcsv(fn, addfn)
        print(f'/Users/me/Downloads/{addfn} was created.')
    except:
        print('usage: python -m studcsv <readfile> [<writefile>]')
        print('<writefile> is created in /Users/me/Downloads folder.')
        
