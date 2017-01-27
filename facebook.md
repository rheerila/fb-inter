88. merge sorted array

```cpp

class Solution {
public:
    void merge(int A[], int m, int B[], int n) {
        
        int a=m-1;
        int b=n-1;
        int i=m+n-1;    // calculate the index of the last element of the merged array
        
        // go from the back by A and B and compare and put to the A element which is larger
        while(a>=0 && b>=0){
            if(A[a]>B[b])   A[i--]=A[a--];
            else            A[i--]=B[b--];
        }
        
        // if B is longer than A just copy the rest of B to A location, otherwise no need to do anything
        while(b>=0)         A[i--]=B[b--];
    }
};


```

21. merge two linked list

```cpp
ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
    
    if(NULL == l1) return l2;
    if(NULL == l2) return l1;
    
    ListNode* head=NULL;    // head of the list to return
    
    // find first element (can use dummy node to put this part inside of the loop)
    if(l1->val < l2->val)       { head = l1; l1 = l1->next; }
    else                        { head = l2; l2 = l2->next; }
    
    ListNode* p = head;     // pointer to form new list
    
    // I use && to remove extra IF from the loop
    while(l1 && l2){
        if(l1->val < l2->val)   { p->next = l1; l1 = l1->next; }
        else                    { p->next = l2; l2 = l2->next; }
        p=p->next;
    }
    
    // add the rest of the tail, done!
    if(l1)  p->next=l1;
    else    p->next=l2;
    
    return head;
}


```

23. merge kth sorted lists

```cpp
struct compare {
    bool operator()(const ListNode* l, const ListNode* r) {
        return l->val > r->val;
    }
};
ListNode *mergeKLists(vector<ListNode *> &lists) { //priority_queue
    priority_queue<ListNode *, vector<ListNode *>, compare> q;
    for(auto l : lists) {
        if(l)  q.push(l);
    }
    if(q.empty())  return NULL;

    ListNode* result = q.top();
    q.pop();
    if(result->next) q.push(result->next);
    ListNode* tail = result;            
    while(!q.empty()) {
        tail->next = q.top();
        q.pop();
        tail = tail->next;
        if(tail->next) q.push(tail->next);
    }
    return result;
}



```

28. Implement strStr()   omn

```cpp
class Solution {
public:
    int strStr(string haystack, string needle) {
        int cnt1 = haystack.length();
        int cnt2 = needle.length();
        if(!cnt2) return 0;

        for(int i = 0 ; i< cnt1-cnt2+1 ; ++i){
            for(int j = 0; j<cnt2 ; j++){
                if(haystack[i+j] != needle[j]) break;
                if(j == cnt2 -1) return i;
            }
            
        }
        return -1;
    }
};

on
class Solution {
public:
    int strStr(string haystack, string needle) {
        if(needle.empty()) return 0;
        if(haystack.empty()) return -1;
        vector<int> pi(needle.size(), 0);
        //KMP-algorithm:
        //Pre-process
        int k = 0, i;
        for(i = 1; i < needle.size(); i++) {
            while(k > 0  && needle[k] != needle[i]) k = pi[k - 1];
            if(needle[k] == needle[i]) pi[i] = ++k;
        }
        k = 0;
        //Matching
        for(i = 0; i < haystack.size(); i++) {
            while(k > 0 && haystack[i] != needle[k]) k = pi[k - 1];
            if(haystack[i] == needle[k]) k++;
            if(k == needle.size()) return i - needle.size() + 1;
        }
        return -1;
    }
};


```



57. Insert Interval
```cpp
vector<Interval> insert(vector<Interval>& intervals, Interval newInterval) {
    vector<Interval> ret;
    auto it = intervals.begin();
    for(; it!=intervals.end(); ++it){
		if(newInterval.end < (*it).start) //all intervals after will not overlap with the newInterval
			break; 
		else if(newInterval.start > (*it).end) //*it will not overlap with the newInterval
			ret.push_back(*it); 
        else{ //update newInterval bacause *it overlap with the newInterval
            newInterval.start = min(newInterval.start, (*it).start);
            newInterval.end = max(newInterval.end, (*it).end);
        }	
    }
    // don't forget the rest of the intervals and the newInterval
	ret.push_back(newInterval);
	for(; it!=intervals.end(); ++it)
		ret.push_back(*it);
	return ret;
}

```



75. sort color on

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int p1= 0,p2 = 0,p3 = 0;
        for(int i=0;i<nums.size();i++){
            if(nums[i] == 0){
                swap(nums[i],nums[p2]);
                swap(nums[p2],nums[p1]);
                p1++;
                p2++
            }
            if(nums[i]  == 1){
                swap(nums[i],nums[p2]);
                p2++;
            }

        }
    }
};

olgn

class Solution {
    public:


    bool Comp(const Interval &a, const Interval &b){
    return a.start<b.start;
    };

    void sortColors(vector<int>& nums) 
    {
        int tmp = 0, low = 0, mid = 0, high = nums.size() - 1;
    
        while(mid <= high)
        {
            if(nums[mid] == 0)
            {
                tmp = nums[low];
                nums[low] = nums[mid];
                nums[mid] = tmp;
                low++;
                mid++;
            }
            else if(nums[mid] == 1)
            {
                mid++;
            }
            else if(nums[mid] == 2)
            {
                tmp = nums[high];
                nums[high] = nums[mid];
                nums[mid] = tmp;
                high--;
            }
        }
    }
};

```


56. merge Interval o(nlogn)
```cpp
class Solution {
public:
    vector<Interval> merge(vector<Interval>& ins) {
    if (ins.empty()) return vector<Interval>{};
    vector<Interval> res;
    sort(ins.begin(), ins.end(), [](Interval a, Interval b){return a.start < b.start;});
    res.push_back(ins[0]);
    for (int i = 1; i < ins.size(); i++) {
        if (res.back().end < ins[i].start) res.push_back(ins[i]);
        else
            res.back().end = max(res.back().end, ins[i].end);
    }
    return res;
}
};

```


67. add binary on2

```cpp
string addBinary(string a, string b) {
        string res ="";
        for(int carry=0, i=a.length()-1, j=b.length()-1; i>=0||j>=0||carry>0;carry/=2)
        {
            if(i>=0) carry+=a[i--]-'0';
            if(j>=0) carry+=b[j--]-'0';
            res = to_string(carry%2)+res;
        }
        return res;
    }


class Solution
{
public:
    string addBinary(string a, string b)
    {
        string s = "";
        
        int c = 0, i = a.size() - 1, j = b.size() - 1;
        while(i >= 0 || j >= 0 || c == 1)
        {
            c += i >= 0 ? a[i --] - '0' : 0;
            c += j >= 0 ? b[j --] - '0' : 0;
            s = char(c % 2 + '0') + s;
            c /= 2;
        }
        
        return s;
    }
};

on
struct Solution {
    string addBinary(string a, string b) {
        if (a.size() < b.size())
            swap(a, b);
        int i = a.size(), j = b.size();
        while (i--) {
            if (j) a[i] += b[--j] & 1;
            if (a[i] > '1') {
                a[i] -= 2;
                if (i) a[i-1]++; else a = '1' + a;
            }
        }
        return a;
    }
};

78. Subsets

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> sub;
        //sort(nums.begin(),nums.end());
        dfs(nums,0,sub,res);
        return res;
    }
    void dfs(vector<int>& nums,int cnt, vector<int>& sub , vector<vector<int>>& res){
        res.push_back(sub);
        for(int i = cnt; i< nums.size() ; i++){
            sub.push_back(nums[i]);
            dfs(nums,i+1,sub,res);
            sub.pop_back();
        }
    }
};
```



79. Word Search m*n*l
```cpp
class Solution {
public:
    
    bool dfs(vector<vector<char>>& board, string &word, int &m, int &n,int &length, int cnt, int row, int col ){
        char c = board[row][col];
        if(board[row][col] != word[cnt]) return false;
        if(board[row][col] == word[cnt] && cnt == length - 1) return true;
        board[row][col] =  '*';
        if(row > 0 ) {
           if(dfs(board,word,m,n,length,cnt+1,row-1,col)) return true;
        }
        if(row < m-1){
           if(dfs(board,word,m,n,length,cnt+1,row+1,col)) return true;        
        }
        if(col>0){
           if(dfs(board,word,m,n,length,cnt+1,row,col-1)) return true;    
        }
        if(col < n-1){
           if(dfs(board,word,m,n,length,cnt+1,row,col+1)) return true;    
        }
        board[row][col] =  c;
        return false;
    }
    
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size();
        int n = board[0].size();
        int length = word.length();
        for(int i = 0; i < m ;i++){
            for(int j = 0 ; j<n ; j++){
                if(dfs(board,word,m,n,length,0,i,j)) return true ;
            }
        }
        return false;
    }
    
};






```







```

91.  Decode Ways
给你一串数字，解码成英文字母。
类似爬楼梯问题，但要加很多限制条件。
定义数组number，number[i]意味着：字符串s[0..i-1]可以有number[i]种解码方法。
回想爬楼梯问题一样，number[i] = number[i-1] + number[i-2]
但不同的是本题有多种限制：
第一： s[i-1]不能是0，如果s[i-1]是0的话，number[i]就只能等于number[i-2]
第二，s[i-2,i-1]中的第一个字符不能是0，而且Integer.parseInt(s.substring(i-2,i))获得的整数必须在0到26之间。

假设解码函数为h。对于一位数X，只能解码成h[X]。而对于一个两位数XY：
1. 如果XY<=26，那么能解码成h[X], h[Y], h[XY]
2. 否则，只能解码成h[X], h[Y]
由于只要求计算最多的解码方法而并不要求每种解码的结果，所以用DP做更为合适高效。

定义dp[i+1]为能解码长度为i+1的string s[0:i]的方法数：
1. dp[0] = 1，dp[1] = 0
2. v = s[i-1]*10+s[i]：
v<=26： dp[i+1] = dp[i] + dp[i-1]
v>26：dp[i+1] = dp[i]


```cpp

class Solution {
public:
    int numDecodings(string s) {
        if (s.empty() || (s.size() > 1 && s[0] == '0')) return 0;
        vector<int> dp(s.size() + 1, 0);
        dp[0] = 1;
        for (int i = 1; i < dp.size(); ++i) {
            dp[i] = (s[i - 1] == '0') ? 0 : dp[i - 1];
            if (i > 1 && (s[i - 2] == '1' || (s[i - 2] == '2' && s[i - 1] <= '6'))) {
                dp[i] += dp[i - 2];
            }
        }
        return dp.back();
    }
};


o(1) Solution

int numDecodings(string s) {
    // empty string or leading zero means no way
    if (!s.size() || s.front() == '0') return 0;

    // r1 and r2 store ways of the last and the last of the last
    int r1 = 1, r2 = 1;

    for (int i = 1; i < s.size(); i++) {
        // zero voids ways of the last because zero cannot be used separately
        if (s[i] == '0') r1 = 0;
        
        // possible two-digit letter, so new r1 is sum of both while new r2 is the old r1
        if (s[i - 1] == '1' || s[i - 1] == '2' && s[i] <= '6') {
            r1 = r2 + r1;
            r2 = r1 - r2;
        }

        // one-digit letter, no new way added
        else {
            r2 = r1;
        }
    }

    return r1;
}
```

98. Validate Binary Search Tree

```cpp
bool isValidBST(TreeNode* root) {
    return isValidBST(root, NULL, NULL);
}

bool isValidBST(TreeNode* root, TreeNode* minNode, TreeNode* maxNode) {
    if(!root) return true;
    if(minNode && root->val <= minNode->val || maxNode && root->val >= maxNode->val)
        return false;
    return isValidBST(root->left, minNode, root) && isValidBST(root->right, root, maxNode);
}

```




125. Valid Palindrome

```cpp

class Solution {
public:
    bool isPalindrome(string s) {
        for (int i = 0, j = s.size() - 1; i < j; i++, j--) { // Move 2 pointers from each end until they collide
        while (isalnum(s[i]) == false && i < j) i++; // Increment left pointer if not alphanumeric
        while (isalnum(s[j]) == false && i < j) j--; // Decrement right pointer if no alphanumeric
        if (toupper(s[i]) != toupper(s[j])) return false; // Exit and return error if not match
    }
    
    return true;
    }
};


```



127. word ladder 26*L n*w

```cpp
class Solution {
public:
    int ladderLength(string beginWord, string endWord, unordered_set<string>& wordList) {
        unordered_map<string, int> dis;
        queue<string> q;
        dis[beginWord] == 1;
        q.push(beginWord);
        while(!q.empty()){
            string word = q.front();
            q.pop();
             if(word == endWord) break;
            for(int i = 0 ; i < word.size();++i){
                string tmp = word;
               
                for(char c = 'a' ; c <='z' ; ++c){
                    tmp[i] = c;
                    if(wordList.count(tmp) > 0 && dis.count(tmp) == 0){
                        dis[tmp] =  dis[word] + 1;
                        q.push(tmp);
                        
                    }
                }
            }
        }
        
        if (dis.count(endWord) == 0) return 0;
        return dis[endWord];    
            
    }
};
```


139. word break
```cpp
class Solution {
public:
    bool wordBreak(string s, unordered_set<string> &dict) {
        if(dict.size()==0) return false;
        
        vector<bool> dp(s.size()+1,false);
        dp[0]=true;
        
        for(int i=1;i<=s.size();i++)
        {
            for(int j=i-1;j>=0;j--)
            {
                if(dp[j])
                {
                    string word = s.substr(j,i-j);
                    if(dict.find(word)!= dict.end())
                    {
                        dp[i]=true;
                        break; //next i
                    }
                }
            }
        }
        
        return dp[s.size()];
    }
};

```

200. Number of Islands omn

```cpp
class Solution {
public:



    void dfs(vector<vector<char>>& grid, int i, int j){
        if(i<0 || j< 0 || i>grid.size()-1 || j >grid[0].size()-1) return;
        if(grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs(grid,i-1,j);
        dfs(grid,i+1,j);
        dfs(grid,i,j-1);
        dfs(grid,i,j+1);
    }
    int numIslands(vector<vector<char>>& grid) {
        int cnt = 0;
        if(grid.size() == 0) return 0;
        for(int i=0;i<=grid.size()-1;i++){
            for(int j = 0;j<=grid[0].size()-1;j++){
                if(grid[i][j] == '1'){
                    cnt++;
                    dfs(grid,i,j);
                }
            }
        }
        return cnt;
    }
};


```


252. meeting room

```cpp

class Solution {
public:
    bool canAttendMeetings(vector<Interval>& intervals) {
        sort(intervals.begin(), intervals.end(), compare);
        int n = intervals.size();
        for (int i = 0; i < n - 1; i++)
            if (overlap(intervals[i], intervals[i + 1]))
                return false;
        return true;
    }
private:
    static bool compare(Interval& interval1, Interval& interval2) {
        return interval1.start < interval2.start;
    }
    bool overlap(Interval& interval1, Interval& interval2) {
        return interval1.end > interval2.start;
    } 
};


sort(intervals.begin(), intervals.end(), 
[](const Interval& a, const Interval& b){ return a.start < b.start; });

```

253. meeting room 2

```cpp
class Solution {
public:
int minMeetingRooms(vector<Interval>& intervals) {
map<int, int> mp; // key: time; val: +1 if start, -1 if end

    for(int i=0; i< intervals.size(); i++) {
        mp[intervals[i].start] ++;
        mp[intervals[i].end] --;
    }
    
    int cnt = 0, maxCnt = 0;
    for(auto it = mp.begin(); it != mp.end(); it++) {
        cnt += it->second;
        maxCnt = max( cnt, maxCnt);
    }
    
    return maxCnt;
}

\\\\\\\\\\\\\\\\\\\\\\\\\\\\

class Solution {
public:
    int minMeetingRooms(vector<Interval>& intervals) {
    map<int, int> changes;
    for (auto i : intervals) {
        changes[i.start] += 1;
        changes[i.end] -= 1;
    }
    int rooms = 0, maxrooms = 0;
    for (auto change : changes)
        maxrooms = max(maxrooms, rooms += change.second);
    return maxrooms;
}
};


```

278. bad version ologn o1

```cpp

class Solution {
public:
    int firstBadVersion(int n) {
        int lower = 1, upper = n, mid;
        while(lower < upper) {
            mid = lower + (upper - lower) / 2;
            if(!isBadVersion(mid)) lower = mid + 1;   /* Only one call to API */
            else upper = mid;
        }
        return lower;   /* Because there will alway be a bad version, return lower here */
    }
};

```

283. Move Zeros on

```cpp
class Solution {
public:
    /*void moveZeroes(vector<int>& nums) {
        int zero = 0, walk = 0, N = nums.size();
        while (walk < N) {
            if (nums[walk++] != 0) 
                swap(nums[zero++], nums[walk - 1]);
        }
    }
    */
    void moveZeroes(vector<int>& nums) {
        int len = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] != 0) {
                if (nums[len] != nums[i]) {
                    swap(nums[len], nums[i]);
                }
                ++len;
            }
        }
    }
};


```


100000. plus Kth nearest pointer

```cpp
struct Point { 
    double x;
    double y; 
    Point(double a, double b) {
        x = a;
        y = b;
    }
};

double getDistance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
typedef bool (*comp)(Point, Point);
Point global_origin = Point(0,0);
bool compare(Point a, Point b)
{
   return (getDistance(a, global_origin)< getDistance(b, global_origin));
}

vector<Point> Solution(vector<Point> &array, Point origin, int k) {
    global_origin = Point(origin.x, origin.y);
    priority_queue<Point, std::vector<Point>, comp> pq(compare);
    vector<Point> ret;
    for (int i = 0; i < array.size(); i++) {
        Point p = array[i];
        pq.push(p);
        if (pq.size() > k)
            pq.pop();
    }
    int index = 0;
    while (!pq.empty()){
        Point p = pq.top();
        ret.push_back(p);
        pq.pop();
    }
    return ret;
}



int main()
{
   Point p1 = Point(4.5, 6.0);
   Point p2 = Point(4.0, 7.0);
   Point p3 = Point(4.0, 4.0);
   Point p4 = Point(2.0, 5.0);
   Point p5 = Point(1.0, 1.0);
   vector<Point> array = {p1, p2, p3, p4, p5};
   int k = 2;
   Point origin = Point(0.0, 0.0);
   vector<Point> ans = Solution(array, origin, k);
   for (int i = 0; i < ans.size(); i++) {
       cout << i << ": " << ans[i].x << "," << ans[i].y << endl;
   }
   //cout << getDistance(p1, p2) << endl;
}




```


