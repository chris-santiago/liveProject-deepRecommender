(self.webpackChunkliveVideoAndProject=self.webpackChunkliveVideoAndProject||[]).push([[149],{15137:(t,e,n)=>{"use strict";function r(t,e){return r=Object.setPrototypeOf||function(t,e){return t.__proto__=e,t},r(t,e)}function o(t,e){t.prototype=Object.create(e.prototype),t.prototype.constructor=t,r(t,e)}n.d(e,{NL:()=>y,l_:()=>C,AW:()=>M,F0:()=>d,rs:()=>j,s6:()=>m,LX:()=>O,k6:()=>k});var i=n(67294),a=(n(45697),n(4852)),p=n(29300),u=n(2177);function c(){return c=Object.assign||function(t){for(var e=1;e<arguments.length;e++){var n=arguments[e];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(t[r]=n[r])}return t},c.apply(this,arguments)}var s=n(39658),l=n.n(s);n(59864),n(28420);var f=function(t){var e=(0,p.Z)();return e.displayName=t,e},h=f("Router-History"),m=f("Router"),d=function(t){function e(e){var n;return(n=t.call(this,e)||this).state={location:e.history.location},n._isMounted=!1,n._pendingLocation=null,e.staticContext||(n.unlisten=e.history.listen((function(t){n._isMounted?n.setState({location:t}):n._pendingLocation=t}))),n}o(e,t),e.computeRootMatch=function(t){return{path:"/",url:"/",params:{},isExact:"/"===t}};var n=e.prototype;return n.componentDidMount=function(){this._isMounted=!0,this._pendingLocation&&this.setState({location:this._pendingLocation})},n.componentWillUnmount=function(){this.unlisten&&(this.unlisten(),this._isMounted=!1,this._pendingLocation=null)},n.render=function(){return i.createElement(m.Provider,{value:{history:this.props.history,location:this.state.location,match:e.computeRootMatch(this.state.location.pathname),staticContext:this.props.staticContext}},i.createElement(h.Provider,{children:this.props.children||null,value:this.props.history}))},e}(i.Component);i.Component;var v=function(t){function e(){return t.apply(this,arguments)||this}o(e,t);var n=e.prototype;return n.componentDidMount=function(){this.props.onMount&&this.props.onMount.call(this,this)},n.componentDidUpdate=function(t){this.props.onUpdate&&this.props.onUpdate.call(this,this,t)},n.componentWillUnmount=function(){this.props.onUnmount&&this.props.onUnmount.call(this,this)},n.render=function(){return null},e}(i.Component);function y(t){var e=t.message,n=t.when,r=void 0===n||n;return i.createElement(m.Consumer,null,(function(t){if(t||(0,u.Z)(!1),!r||t.staticContext)return null;var n=t.history.block;return i.createElement(v,{onMount:function(t){t.release=n(e)},onUpdate:function(t,r){r.message!==e&&(t.release(),t.release=n(e))},onUnmount:function(t){t.release()},message:e})}))}var g={},x=0;function E(t,e){return void 0===t&&(t="/"),void 0===e&&(e={}),"/"===t?t:function(t){if(g[t])return g[t];var e=l().compile(t);return x<1e4&&(g[t]=e,x++),e}(t)(e,{pretty:!0})}function C(t){var e=t.computedMatch,n=t.to,r=t.push,o=void 0!==r&&r;return i.createElement(m.Consumer,null,(function(t){t||(0,u.Z)(!1);var r=t.history,p=t.staticContext,s=o?r.push:r.replace,l=(0,a.ob)(e?"string"==typeof n?E(n,e.params):c({},n,{pathname:E(n.pathname,e.params)}):n);return p?(s(l),null):i.createElement(v,{onMount:function(){s(l)},onUpdate:function(t,e){var n=(0,a.ob)(e.to);(0,a.Hp)(n,c({},l,{key:n.key}))||s(l)},to:n})}))}var b={},w=0;function O(t,e){void 0===e&&(e={}),("string"==typeof e||Array.isArray(e))&&(e={path:e});var n=e,r=n.path,o=n.exact,i=void 0!==o&&o,a=n.strict,p=void 0!==a&&a,u=n.sensitive,c=void 0!==u&&u;return[].concat(r).reduce((function(e,n){if(!n&&""!==n)return null;if(e)return e;var r=function(t,e){var n=""+e.end+e.strict+e.sensitive,r=b[n]||(b[n]={});if(r[t])return r[t];var o=[],i={regexp:l()(t,o,e),keys:o};return w<1e4&&(r[t]=i,w++),i}(n,{end:i,strict:p,sensitive:c}),o=r.regexp,a=r.keys,u=o.exec(t);if(!u)return null;var s=u[0],f=u.slice(1),h=t===s;return i&&!h?null:{path:n,url:"/"===n&&""===s?"/":s,isExact:h,params:a.reduce((function(t,e,n){return t[e.name]=f[n],t}),{})}}),null)}var M=function(t){function e(){return t.apply(this,arguments)||this}return o(e,t),e.prototype.render=function(){var t=this;return i.createElement(m.Consumer,null,(function(e){e||(0,u.Z)(!1);var n=t.props.location||e.location,r=c({},e,{location:n,match:t.props.computedMatch?t.props.computedMatch:t.props.path?O(n.pathname,t.props):e.match}),o=t.props,a=o.children,p=o.component,s=o.render;return Array.isArray(a)&&function(t){return 0===i.Children.count(t)}(a)&&(a=null),i.createElement(m.Provider,{value:r},r.match?a?"function"==typeof a?a(r):a:p?i.createElement(p,r):s?s(r):null:"function"==typeof a?a(r):null)}))},e}(i.Component);i.Component;var j=function(t){function e(){return t.apply(this,arguments)||this}return o(e,t),e.prototype.render=function(){var t=this;return i.createElement(m.Consumer,null,(function(e){e||(0,u.Z)(!1);var n,r,o=t.props.location||e.location;return i.Children.forEach(t.props.children,(function(t){if(null==r&&i.isValidElement(t)){n=t;var a=t.props.path||t.props.from;r=a?O(o.pathname,c({},t.props,{path:a})):e.match}})),r?i.cloneElement(n,{location:o,computedMatch:r}):null}))},e}(i.Component),P=i.useContext;function k(){return P(h)}},28420:(t,e,n)=>{"use strict";var r=n(59864),o={childContextTypes:!0,contextType:!0,contextTypes:!0,defaultProps:!0,displayName:!0,getDefaultProps:!0,getDerivedStateFromError:!0,getDerivedStateFromProps:!0,mixins:!0,propTypes:!0,type:!0},i={name:!0,length:!0,prototype:!0,caller:!0,callee:!0,arguments:!0,arity:!0},a={$$typeof:!0,compare:!0,defaultProps:!0,displayName:!0,propTypes:!0,type:!0},p={};function u(t){return r.isMemo(t)?a:p[t.$$typeof]||o}p[r.ForwardRef]={$$typeof:!0,render:!0,defaultProps:!0,displayName:!0,propTypes:!0},p[r.Memo]=a;var c=Object.defineProperty,s=Object.getOwnPropertyNames,l=Object.getOwnPropertySymbols,f=Object.getOwnPropertyDescriptor,h=Object.getPrototypeOf,m=Object.prototype;t.exports=function t(e,n,r){if("string"!=typeof n){if(m){var o=h(n);o&&o!==m&&t(e,o,r)}var a=s(n);l&&(a=a.concat(l(n)));for(var p=u(e),d=u(n),v=0;v<a.length;++v){var y=a[v];if(!(i[y]||r&&r[y]||d&&d[y]||p&&p[y])){var g=f(n,y);try{c(e,y,g)}catch(t){}}}}return e}},76585:t=>{t.exports=Array.isArray||function(t){return"[object Array]"==Object.prototype.toString.call(t)}},39658:(t,e,n)=>{var r=n(76585);t.exports=function t(e,n,o){return r(n)||(o=n||o,n=[]),o=o||{},e instanceof RegExp?function(t,e){var n=t.source.match(/\((?!\?)/g);if(n)for(var r=0;r<n.length;r++)e.push({name:r,prefix:null,delimiter:null,optional:!1,repeat:!1,partial:!1,asterisk:!1,pattern:null});return s(t,e)}(e,n):r(e)?function(e,n,r){for(var o=[],i=0;i<e.length;i++)o.push(t(e[i],n,r).source);return s(new RegExp("(?:"+o.join("|")+")",l(r)),n)}(e,n,o):function(t,e,n){return f(i(t,n),e,n)}(e,n,o)},t.exports.parse=i,t.exports.compile=function(t,e){return p(i(t,e),e)},t.exports.tokensToFunction=p,t.exports.tokensToRegExp=f;var o=new RegExp(["(\\\\.)","([\\/.])?(?:(?:\\:(\\w+)(?:\\(((?:\\\\.|[^\\\\()])+)\\))?|\\(((?:\\\\.|[^\\\\()])+)\\))([+*?])?|(\\*))"].join("|"),"g");function i(t,e){for(var n,r=[],i=0,a=0,p="",s=e&&e.delimiter||"/";null!=(n=o.exec(t));){var l=n[0],f=n[1],h=n.index;if(p+=t.slice(a,h),a=h+l.length,f)p+=f[1];else{var m=t[a],d=n[2],v=n[3],y=n[4],g=n[5],x=n[6],E=n[7];p&&(r.push(p),p="");var C=null!=d&&null!=m&&m!==d,b="+"===x||"*"===x,w="?"===x||"*"===x,O=n[2]||s,M=y||g;r.push({name:v||i++,prefix:d||"",delimiter:O,optional:w,repeat:b,partial:C,asterisk:!!E,pattern:M?c(M):E?".*":"[^"+u(O)+"]+?"})}}return a<t.length&&(p+=t.substr(a)),p&&r.push(p),r}function a(t){return encodeURI(t).replace(/[\/?#]/g,(function(t){return"%"+t.charCodeAt(0).toString(16).toUpperCase()}))}function p(t,e){for(var n=new Array(t.length),o=0;o<t.length;o++)"object"==typeof t[o]&&(n[o]=new RegExp("^(?:"+t[o].pattern+")$",l(e)));return function(e,o){for(var i="",p=e||{},u=(o||{}).pretty?a:encodeURIComponent,c=0;c<t.length;c++){var s=t[c];if("string"!=typeof s){var l,f=p[s.name];if(null==f){if(s.optional){s.partial&&(i+=s.prefix);continue}throw new TypeError('Expected "'+s.name+'" to be defined')}if(r(f)){if(!s.repeat)throw new TypeError('Expected "'+s.name+'" to not repeat, but received `'+JSON.stringify(f)+"`");if(0===f.length){if(s.optional)continue;throw new TypeError('Expected "'+s.name+'" to not be empty')}for(var h=0;h<f.length;h++){if(l=u(f[h]),!n[c].test(l))throw new TypeError('Expected all "'+s.name+'" to match "'+s.pattern+'", but received `'+JSON.stringify(l)+"`");i+=(0===h?s.prefix:s.delimiter)+l}}else{if(l=s.asterisk?encodeURI(f).replace(/[?#]/g,(function(t){return"%"+t.charCodeAt(0).toString(16).toUpperCase()})):u(f),!n[c].test(l))throw new TypeError('Expected "'+s.name+'" to match "'+s.pattern+'", but received "'+l+'"');i+=s.prefix+l}}else i+=s}return i}}function u(t){return t.replace(/([.+*?=^!:${}()[\]|\/\\])/g,"\\$1")}function c(t){return t.replace(/([=!:$\/()])/g,"\\$1")}function s(t,e){return t.keys=e,t}function l(t){return t&&t.sensitive?"":"i"}function f(t,e,n){r(e)||(n=e||n,e=[]);for(var o=(n=n||{}).strict,i=!1!==n.end,a="",p=0;p<t.length;p++){var c=t[p];if("string"==typeof c)a+=u(c);else{var f=u(c.prefix),h="(?:"+c.pattern+")";e.push(c),c.repeat&&(h+="(?:"+f+h+")*"),a+=h=c.optional?c.partial?f+"("+h+")?":"(?:"+f+"("+h+"))?":f+"("+h+")"}}var m=u(n.delimiter||"/"),d=a.slice(-m.length)===m;return o||(a=(d?a.slice(0,-m.length):a)+"(?:"+m+"(?=$))?"),a+=i?"$":o&&d?"":"(?="+m+"|$)",s(new RegExp("^"+a,l(n)),e)}}}]);