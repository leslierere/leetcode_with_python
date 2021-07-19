public static int countTeams(int num, int[] skills, int minAssociates, int minLevel, int maxLevel){
        if (skills == null || skills.length == 0) {
            return 0;
        }
        int counts = 0;
        for (int skill : skills) {
            if (skill >= minLevel && skill <= maxLevel) {
                counts ++;
            }
        }
        if (counts < minAssociates) {
            return 0;
        }
        int[][] memo = new int[21][21];
        int res = 0;
        for (int i = minAssociates; i <= counts; i++) {
            res += comb(i, counts, memo);
        }
        return res;
    }
    
    private static int comb(int n, int k, int[][] memo) {
        if (n == 0) {
            return 1;
        }
        if (n == k || k == 0) {
            return 1;
        }
        if (n == k-1 || k == 1) {
            return k;
        }
        if (n < k * 2) {
            return comb(n, n-k, memo);
        }
        if (memo[n][k] > 0) {
            return memo[n][k];
        }
        memo[n][k] = comb(n-1, k-1, memo) + comb(n-1, k, memo);
        return memo[n][k];
    }