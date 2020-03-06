# coding=utf-8
# Copyright 2020 Heewon Jeon. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def sent_to_spacing_chars(sent):
    # 공백은 ^
    chars = sent.strip().replace(' ', '^')
    # char_list = [li.strip().replace(' ', '^') for li in sents]

    # 문장의 시작 포인트 «
    # 문장의 끌 포인트  »
    tagged_chars = "«" + chars + "»"
    # char_list = [ "«" + li + "»" for li in char_list]

    # 문장 -> 문자열
    char_list = ' '.join(list(tagged_chars))
    # char_list = [ ' '.join(list(li))  for li in char_list]
    return(char_list)
